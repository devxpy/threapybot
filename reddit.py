import json
import random
from multiprocessing.pool import ThreadPool
from time import sleep

import feedparser
import pandas as pd
import requests
from pyquery import PyQuery

# from https://useragentstring.com/
FAKE_USER_AGENTS = [
    # chrome
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.79 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.53 Safari/537.36",
    # edge
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19577",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/18.17720",
    "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.8810.3391 Safari/537.36 Edge/18.14383",
]

ret = []


def main():
    next_url = "https://www.reddit.com/r/mentalhealth/top.rss?t=year&limit=100"

    for _ in range(10):
        r = requests.get(
            next_url,
            headers={"User-Agent": random.choice(FAKE_USER_AGENTS)},
        )
        r.raise_for_status()
        feed = feedparser.parse(r.content)

        # with open("out.xml", "wb") as f:
        #     f.write(r.content)

        # with open("out.xml", "rb") as f:
        #     feed = feedparser.parse(f)

        if not feed.entries:
            return
        for entry in feed.entries:
            print(entry.id, entry.link)
            # pool.apply(get_transcripts, args=[entry.link])
            get_transcripts(entry.link)

        next_url = f"https://www.reddit.com/r/mentalhealth/top.rss?t=year&limit=100&before={feed.entries[-1].id}"

        sleep(200)


def get_transcripts(link):
    r = requests.get(
        link + ".rss?depth=1&limit=10",
        headers={"User-Agent": random.choice(FAKE_USER_AGENTS)},
    )
    r.raise_for_status()
    feed = feedparser.parse(r.content)
    try:
        html = (
            feed.entries[0]
            .content[0]
            .value.split("<!-- SC_OFF -->")[1]
            .split("<!-- SC_ON -->")[0]
        )
    except IndexError:
        return
    prompt = PyQuery(html.encode()).text()
    prompt = " ".join(feed.feed["title"].split(":")[:-1]) + "\n" + prompt

    for entry in feed.entries[1:]:
        try:
            if entry.author == feed.entries[0].author:
                continue
        except AttributeError:
            continue
        html = (
            entry.content[0]
            .value.split("<!-- SC_OFF -->")[1]
            .split("<!-- SC_ON -->")[0]
        )
        transcript = PyQuery(html.encode()).text()
        if "[deleted]" in transcript:
            continue
        ret.append([link, prompt, transcript])

    with open("out.json", "w") as f:
        json.dump(ret, f, indent=2)


with ThreadPool(100) as pool:
    main()

with open("out.json") as f:
    ret = json.load(f)
    df = pd.DataFrame(ret, columns=["source", "title", "transcript"])
    df.sort_values(["source", "title"], inplace=True)
    df.to_csv("out.csv", index=False)
