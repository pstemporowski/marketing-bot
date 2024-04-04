import requests
import dotenv
import os
from bs4 import BeautifulSoup
from openai import OpenAI
import pandas as pd
from urllib.parse import urlparse
from utils.logger import logger
import time
import locale

locale.setlocale(locale.LC_TIME, "de_DE")

dotenv.load_dotenv()
GPT_MODEL = os.environ["GPT_MODEL"]
GPT_API_KEY = os.environ["GPT_API_KEY"]
GPT_MAX_TOKENS = int(os.environ["GPT_MAX_TOKENS"])
CITY = "munchen"
DATA_DIR = f"./data/{CITY}/"
PARTIES_COLS = ["title", "genre", "club_name", "time"]
URLS_COLS = ["url"]
WRITE_PARTIES_PER_WEEK = True

gpt = OpenAI(
    api_key=GPT_API_KEY,  # this is also the default, it can be omitted
)


def main():
    # URL of the page to fetch and parse
    parties_path = os.path.join(DATA_DIR, "parties.csv")
    temp_path = os.path.join(DATA_DIR, "temp_1.csv")
    urls_path = os.path.join(DATA_DIR, "urls.csv")
    urls = pd.read_csv(urls_path)
    parties_df = None

    if os.path.exists(parties_path):
        parties_df = pd.read_csv(parties_path, header=0)
    else:
        parties_df = pd.DataFrame(columns=PARTIES_COLS)

    for i, (url, mode, split_cls, club_name, genre) in urls.iterrows():
        df = get_parties(
            url,
            mode,
            split_cls,
            dft_club_name=club_name,
            dft_genre=genre,
        )
        parties_df = pd.concat([parties_df, df])

        if i % 5 == 0 and i != 0:
            parties_df.to_csv(temp_path, index=False)

    parties_df = parties_df.drop_duplicates(keep="first")
    parties_df.to_csv(parties_path, index=False)

    if WRITE_PARTIES_PER_WEEK:
        write_parties_per_week(parties_df)


def get_parties(
    url: str, mode="default", split_cls_name=None, dft_club_name=None, dft_genre=None
):
    # Fetch and parse the HTML content from the URL of the events page
    html = get_html(url)

    if mode == "default" or mode == "detail" or mode == "links":
        links = get_links_from_overview(html, url)

        if len(links) == 0:
            logger.warn(f"No links found to extract information from on {url}.")
        else:
            logger.info(f"Found {len(links)} links to extract information from.")

        return get_parties_from_detail_pages(links)
    elif mode == "overview":
        return get_parties_from_overview(
            html,
            split_cls_name=split_cls_name,
            dft_club_name=dft_club_name,
            dft_genre=dft_genre,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")


def get_links_from_overview(html, url):
    links_df = ext_links(html, return_pd=True)
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    links_df = clean_links(
        links_df, base_url=base_url, contains_list=["party", "event"]
    )

    return links_df["links"]


def get_parties_from_overview(
    html, split_cls_name=None, dft_club_name=None, dft_genre=None
):
    assert split_cls_name is not None, "split_class_name must be provided."

    parties = []
    soup = BeautifulSoup(html, "html.parser")
    soup = soup.find_all(class_=split_cls_name)
    for i, s in enumerate(soup):
        txt = s.get_text(strip=True)

        try:
            ctx = extr_info_with_llm(txt)
            ctx = ctx.split(";")

            if dft_club_name:
                ctx[2] = dft_club_name

            if dft_genre and ctx[1] == "None" or ctx[1] == "Mixed":
                ctx[1] = dft_genre

            if len(ctx) != 4:
                logger.warning(
                    f"Something went wrong with the extraction of the information for split {i}."
                )
                continue

            parties.append(ctx)
        except Exception as e:
            logger.error(f"Error while extracting information from split {i}: {e}")
            continue

    return pd.DataFrame(parties, columns=PARTIES_COLS)


def get_parties_from_detail_pages(links):
    parties = []

    for link in links:
        try:
            html = get_html(link)
            txt = parse_html_2_txt(html)

            ctx = extr_info_with_llm(txt)
            ctx = ctx.split(";")

            if len(ctx) != 4:
                logger.warning(
                    f"Something went wrong with the extraction of the information for the link: {link}"
                )
                continue

            parties.append(ctx)
        except Exception as e:
            logger.error(f"Error while extracting information from {link}: {e}")

    return pd.DataFrame(parties, columns=PARTIES_COLS)


def extr_info_with_llm(txt: str):
    """
    Extracts information from a given description using the OpenAI API.

    Args:
        txt (str): The description in German from which information needs to be extracted.

    Returns:
        str: The extracted information formatted as "{{Title}};{{Genre}};{{Clubname}};{{Date in YYYY-MM-DD HH:MM:SS}};"

    Example:
        Die beste 90s Hits;Techno;Batschkapp;2022-10-19 20:00:00;
    """
    # Here, you formulate a prompt for the LLM to instruct it on what information to extract.
    prompt = f"""Extract the title, genre, club name, datetime from this description in german: "{txt}". Choose a genre out of the text between Techno, Electronic, Rock, Metal, Hip-Hop, 80s, 90s, Pop and Mixed! The genres are very general so as they is for example PopPunk you can classify it as Pop and so on. As in the text are mentioned more than one genre that cannot be classified to one, like in the text is Electronic and Hip-Hop, so you gonna write Mixed. Also don't write the city name in the title or the club name. Moreover if the title is a sentence or a persons name write it case sensitivity. Try to rewrite the title as much as possible in case sensitive. You need to classify the provided information, if it is actually a party or not. For example there can be Situations where the event is stand up comedy or a 'flohmarkt' and so on, if it is the case whole answer blank!  Don't rewrite the content but you can short it somehow so that the title is no longer then 25 chars and format the output as: 

    "{{Title}};{{Genre}};{{Clubname}};{{Date in YYYY-MM-DD HH:MM:SS}}"
    
    Example:
    Die beste 90s Hits;Techno;Batschkapp;2022-10-19 20:00:00

    Example for a blank answer where the party is flohmarkt:
    ;;;
    
    Example for a blank answer where the party is stand up comedy:
    ;;;
    
    """

    # Call the OpenAI API to get the answer to your question
    res = gpt.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=GPT_MAX_TOKENS,
        n=1,
        stop=None,
        temperature=0.0,
    )

    return res.choices[0].message.content


def get_html(url: str):
    # Fetch HTML content
    response = requests.get(url, timeout=10)
    html = response.text
    return html


def get_links(html):
    # Get all links from the HTML content
    soup = BeautifulSoup(html, "html.parser")
    return soup.find_all("a")


def parse_html_2_txt(html):
    # Parse HTML content
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(strip=True)


def ext_links(content, return_pd=False):
    links = []
    all_links = get_links(content)

    # Get all the links that provide to other detail pages of parties and events
    for link in all_links:
        link = link.get("href")

        if link:
            links.append(link)

    if return_pd:
        return pd.DataFrame(links, columns=["links"])

    return links


def clean_links(links_df, base_url=None, contains_list=None, rm_duplicates=True):
    if base_url:
        # Make all links absolute
        links_df["links"] = links_df["links"].apply(
            lambda link: base_url + link if not is_absolute(link) else link
        )

        # Remove all links that are not base_url
        links_df = links_df[links_df["links"].str.startswith(base_url)]
    else:
        links_df = links_df[~links_df["links"].str.startswith("/")]

    if contains_list:
        links_df = links_df[links_df["links"].str.contains("|".join(contains_list))]

    if rm_duplicates:
        links_df = links_df.drop_duplicates(keep="first")

    return links_df


def is_absolute(url):
    return bool(urlparse(url).netloc)


def write_parties_per_week(parties_df: pd.DataFrame):
    parties_df["time"] = pd.to_datetime(
        parties_df["time"],
        errors="coerce",
        format="mixed",
    )
    parties_df = parties_df.sort_values(by="time")
    parties_df["week"] = parties_df["time"].dt.isocalendar().week
    weeks = parties_df["week"].unique()

    for week in weeks:
        week_df = parties_df[parties_df["week"] == week]
        week_df = week_df.drop(columns=["week"])
        week_txt = ""
        for _, row in week_df.iterrows():
            title = row["title"]
            genre = row["genre"]
            club_name = row["club_name"]
            time = row["time"].strftime("%a. %d.%m")
            hour = row["time"].strftime("%H:%M")
            if genre:
                party_info = f"{title} ({genre})"
            else:
                party_info = title
            party_info += f"\n{club_name} - {time} ab {hour} Uhr\n\n"
            week_txt += party_info
        with open(
            os.path.join(DATA_DIR, f"parties_per_week/week_{week}.txt"), "w"
        ) as file:
            file.write(week_txt)


if __name__ == "__main__":
    main()
