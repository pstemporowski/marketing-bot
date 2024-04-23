from bs4 import BeautifulSoup
from openai import OpenAI
from urllib.parse import urlparse
import dotenv
import locale
import os
import pandas as pd
import re
import requests
from requests.exceptions import ChunkedEncodingError, RequestException

from utils.logger import logger
from datetime import datetime
from datetime import datetime
import argparse

locale.setlocale(locale.LC_TIME, "de_DE")

parser = argparse.ArgumentParser(description="Script to extract data.")
parser.add_argument("-city", type=str, help="City name", default="munchen")
args = parser.parse_args()

dotenv.load_dotenv()
GPT_MODEL = os.environ["GPT_MODEL"]
GPT_API_KEY = os.environ["GPT_API_KEY"]
GPT_MAX_TOKENS = int(os.environ["GPT_MAX_TOKENS"])


CITY = args.city
DATA_DIR = f"./data/{CITY}/"
IMPORTANT_GENRES = ["Hip-Hop", "Rock", "Techno", "Metal"]
PARTIES_COLS = ["title", "genre", "club_name", "time"]
URLS_COLS = ["url"]
WRITE_PARTIES_PER_WEEK = True
FILTER_TITLES = [
    "flohmarkt",
    "bundesliga",
    "stand up comedy",
    "comedy",
    "kabarett",
]


gpt = OpenAI(
    api_key=GPT_API_KEY,  # this is also the default, it can be omitted
)


def main():
    # URL of the page to fetch and parse
    parties_path = os.path.join(DATA_DIR, "parties.csv")
    temp_path = os.path.join(DATA_DIR, "temp_1.csv")
    urls_path = os.path.join(DATA_DIR, "urls.csv")
    manual_entries_path = os.path.join(DATA_DIR, "manual_entries.csv")
    urls = pd.read_csv(urls_path, sep=";", na_values=["None"])
    parties_df = pd.DataFrame(columns=PARTIES_COLS)
    manual_entries_df = pd.read_csv(manual_entries_path, header=0)

    for i, row in urls.iterrows():
        df = get_parties(
            url=row["url"],
            mode=row["mode"],
            split_cls_name=row["split_by_class"],
            filter_regex=row["filter_regex"],
            dft_club_name=row["club_name"],
            dft_genre=row["genre"],
            link_cls=row["link_class"],
        )
        parties_df = pd.concat([parties_df, df])

        if i % 5 == 0 and i != 0:
            parties_df.to_csv(temp_path, index=False)

    manual_entries_df = pre_process_manual_entries(manual_entries_df)
    parties_df = pd.concat([parties_df, manual_entries_df])

    parties_df = post_process_parties(parties_df, filter_titles=FILTER_TITLES)
    parties_df.to_csv(parties_path, index=False)

    if WRITE_PARTIES_PER_WEEK:
        write_parties_per_week(parties_df)


def pre_process_manual_entries(manual_entries_df: pd.DataFrame):
    entries = []
    for _, row in manual_entries_df.iterrows():
        start_date = datetime.now()
        end_date = start_date + pd.DateOffset(months=2)
        delta = pd.DateOffset(days=7)

        new_entries = []
        if pd.isnull(row["repeat"]):
            new_entry = row.copy()
            new_entry.drop("repeat", inplace=True)
            new_entries.append(new_entry)
        else:
            for date in pd.date_range(start_date, end_date, freq=delta):
                new_entry = row.copy()
                new_entry["time"] = date
                new_entry.drop("repeat", inplace=True)
                new_entries.append(new_entry)

        entries.extend(new_entries)
    print(entries)
    return pd.DataFrame(entries, columns=PARTIES_COLS)


def post_process_parties(parties_df: pd.DataFrame, filter_titles=None):
    if filter_titles:
        flrt_re = "|".join(filter_titles)
        fltr_mask = ~parties_df["title"].str.contains(
            flrt_re, case=False, na=False, regex=True
        )
        parties_df = parties_df[fltr_mask]

    parties_df = parties_df.dropna(subset=["club_name", "title"])
    parties_df["club_name"] = parties_df["club_name"].str.title()
    parties_df["title"] = parties_df["title"].str.title()
    parties_df["time"] = pd.to_datetime(
        parties_df["time"],
        format="mixed",
        errors="coerce",
    )
    parties_df = parties_df.drop_duplicates(
        subset=["club_name", "title", "time"], keep="first"
    ).reset_index(drop=True)
    parties_df = parties_df.sort_values(by="time")
    return parties_df


def get_parties(
    url: str,
    mode="default",
    split_cls_name=None,
    filter_regex=None,
    dft_club_name=None,
    dft_genre=None,
    link_cls=None,
):
    # Fetch and parse the HTML content from the URL of the events page
    html = get_html(url)
    if html is None:
        return pd.DataFrame(columns=PARTIES_COLS)

    if mode == "default" or mode == "detail" or mode == "links":
        links = get_links_from_overview(html, url, link_cls=link_cls)

        if len(links) == 0:
            logger.warning(f"No links found to extract information from on {url}.")
        else:
            logger.info(
                f"Found {len(links)} links to extract information from on {url}."
            )

        return get_parties_from_detail_pages(links)
    elif mode == "overview":
        return get_parties_from_overview(
            html,
            split_cls_name=split_cls_name,
            dft_club_name=dft_club_name,
            dft_genre=dft_genre,
            filter_regex=filter_regex,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")


def get_links_from_overview(html, url, link_cls=None):
    links_df = ext_links(html, return_pd=True, link_cls=link_cls)
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    contains_list = None

    if pd.isnull(link_cls):
        contains_list = ["party", "event", "veranstaltung", "programm", "program"]

    links_df = post_process_links(
        links_df,
        base_url=base_url,
        contains_list=contains_list,
    )

    return links_df["links"]


def get_parties_from_overview(
    html,
    split_cls_name=None,
    dft_club_name=None,
    dft_genre=None,
    filter_regex=None,
):
    assert split_cls_name is not None, "split_class_name must be provided."

    parties = []
    soup = BeautifulSoup(html, "html.parser")
    soup = soup.find_all(class_=split_cls_name)
    for i, s in enumerate(soup):

        if not pd.isnull(filter_regex):
            pattern = re.compile(filter_regex)
            if not pattern.search(str(s)):
                logger.info(f"Skipping split because it does not match the filter.")
                continue

        txt = s.get_text(strip=True)

        try:
            ctx = extr_info_with_llm(txt)
            ctx = ctx.split(";")

            if len(ctx) != 4 or pd.isnull(ctx[0]):
                logger.warning(
                    f"Something went wrong with the extraction of the information for split {i}."
                )
                continue

            if not pd.isnull(dft_club_name):
                ctx[2] = dft_club_name

            if not pd.isnull(dft_genre) and ctx[1] == "None" or ctx[1] == "Mixed":
                ctx[1] = dft_genre

            parties.append(ctx)
        except Exception as e:
            logger.error(f"Error while extracting information from split {i}: {e}")
            continue

    return pd.DataFrame(parties, columns=PARTIES_COLS)


def get_parties_from_detail_pages(links, dft_club_name=None, dft_genre=None):
    parties = []

    for link in links:
        try:
            html = get_html(link)
            if html is None:
                continue

            txt = parse_html_2_txt(html)

            ctx = extr_info_with_llm(txt)
            ctx = ctx.split(";")

            if len(ctx) != 4 or pd.isnull(ctx[0]):
                logger.warning(
                    f"Something went wrong with the extraction of the information for the link: {link}"
                )
                continue

            if not pd.isnull(dft_club_name):
                ctx[2] = dft_club_name

            if not pd.isnull(dft_genre) and ctx[1] == "None" or ctx[1] == "Mixed":
                ctx[1] = dft_genre

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
    prompt = f"""Extract the title, genre, club name, and datetime from this German party description: "{txt}". Identify a genre from the options: Techno, Electronic, Rock, Metal, Hip-Hop, 80s, 90s, Pop, and Mixed. If the description includes a genre such as 'PopPunk', classify it under the nearest category, e.g., Pop. If multiple genres are mentioned that don’t align under one category, label it as Mixed. Base your genre choice on the event title and description; if uncertain, leave the genre field blank. Avoid including the city name in the title or club name. Format the title to ensure case sensitivity and shorten it if necessary to no more than 25 characters. Evaluate whether the text describes a party or another type of event like a 'flohmarkt' (flea market) or stand-up comedy, in which case, return a blank response. Output the results in the following format:

        "{{Title}};{{Genre}};{{Clubname}};{{Date in YYYY-MM-DD HH:MM:SS}}"

    Examples:
    - Normal party:
    Die beste 90s Hits;Techno;Batschkapp;2022-10-19 20:00:00
    - Event not a party (e.g., flea market):
    ;;;
    - Non-party event (e.g., stand-up comedy):
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
        temperature=0.15,
    )

    return res.choices[0].message.content


def get_html(url: str) -> str | None:
    # Fetch HTML content
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    for _ in range(5):
        try:

            response = requests.get(
                url,
                timeout=10,
                headers=headers,
            )
            return response.text
        except RequestException:
            continue
        except ChunkedEncodingError:
            continue

    logger.error(f"Failed to fetch HTML content from {url}.")
    return None


def get_links(html, link_cls=None) -> list:
    # Get all links from the HTML content
    links = []
    a_tags = []
    soup = BeautifulSoup(html, "html.parser")

    if pd.isnull(link_cls):
        a_tags = soup.find_all("a")
    else:
        a_tags = soup.find_all("a", class_=link_cls)

        if len(a_tags) == 0:
            elements = soup.find_all(class_=link_cls)
            for e in elements:
                a_tags.extend(e.find_all("a"))

    for a in a_tags:
        link = a.get("href")
        if link:
            links.append(link)

    return links


def parse_html_2_txt(html):
    # Parse HTML content
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(strip=True)


def ext_links(content, return_pd=False, link_cls=None):
    links = get_links(content, link_cls=link_cls)
    # Get all the links that provide to other detail pages of parties and events
    if return_pd:
        return pd.DataFrame(links, columns=["links"])

    return links


def post_process_links(links_df, base_url=None, contains_list=None, rm_duplicates=True):
    if base_url:
        # Make all links absolute
        links_df["links"] = links_df["links"].apply(
            lambda link: base_url + "/" + link if not is_absolute(link) else link
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


def drop_out_parties(parties_df: pd.DataFrame, limit=40):
    filtered_df = parties_df[parties_df["genre"].isin(IMPORTANT_GENRES)]

    if len(filtered_df) < limit:
        other_df = parties_df[~parties_df["genre"].isin(IMPORTANT_GENRES)]
        if len(other_df) > 0:
            random_parties = other_df.sample(
                n=min(limit - len(filtered_df), len(other_df)), replace=True
            )
            filtered_df = pd.concat([filtered_df, random_parties])
    elif len(filtered_df) > limit:
        filtered_df = filtered_df.sample(n=limit, replace=False)

    return filtered_df


def write_parties_per_week(parties_df: pd.DataFrame):
    parties_per_week_dir = os.path.join(DATA_DIR, "parties_per_week")
    os.makedirs(parties_per_week_dir, exist_ok=True)

    parties_df = parties_df[
        parties_df["time"] <= pd.Timestamp.now() + pd.DateOffset(months=6)
    ]
    parties_df = parties_df.sort_values(by="time")
    parties_df["week"] = parties_df["time"].dt.isocalendar().week
    weeks = parties_df["week"].unique()

    for week in weeks:
        week_df = parties_df[parties_df["week"] == week]
        week_df = drop_out_parties(week_df)
        week_df = week_df.drop(columns=["week"])
        week_df = week_df.sort_values(by="time")
        week_txt = ""
        for _, row in week_df.iterrows():
            title = row["title"]
            genre = row["genre"]
            club_name = row["club_name"]
            time = row["time"].strftime("%a. %d.%m")
            hour = (
                row["time"].strftime("%H:%M")
                if row["time"].minute != 0
                else row["time"].strftime("%H")
            )
            if genre:
                party_info = f"{title} ({genre})"
            else:
                party_info = title
            party_info += f"\n{club_name} - {time} ab {hour} Uhr\n\n"
            week_txt += party_info

        parties_per_week_path = os.path.join(parties_per_week_dir, f"week_{week}.txt")
        with open(parties_per_week_path, "w") as file:
            file.write(week_txt)


if __name__ == "__main__":
    main()
