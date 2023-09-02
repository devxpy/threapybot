import json
from multiprocessing.pool import ThreadPool

import pandas as pd
import requests_html as rh

ret = []


def main():
    s = rh.HTMLSession()
    r = s.get("http://www.thetherapist.com/Transcripts.html")
    r.raise_for_status()
    for link in r.html.absolute_links:
        if not link.endswith("Transcripts.html"):
            continue
        pool.apply(get_pages, args=[link])


def get_pages(link):
    s = rh.HTMLSession()
    r = s.get(link)
    r.raise_for_status()
    el = r.html.find("h3", first=True)
    for link in el.absolute_links:
        if not link.endswith(".html"):
            continue
        # pool.submit(get_transcripts, link)
        pool.apply(get_transcripts, args=[link])


def get_transcripts(link):
    try:
        s = rh.HTMLSession()
        r = s.get(link)
        r.raise_for_status()
        title, transcript = r.html.find("table")[1:3]
        ret.append([link, title.text.strip(), transcript.text.strip()])
        with open("out.json", "w") as f:
            json.dump(ret, f, indent=2)
    except Exception as e:
        print(link, e)


with ThreadPool(100) as pool:
    main()

with open("out.json") as f:
    ret = json.load(f)
    df = pd.DataFrame(ret, columns=["source", "title", "transcript"])
    df.sort_values(["source", "title"], inplace=True)
    df.to_csv("out.csv", index=False)
