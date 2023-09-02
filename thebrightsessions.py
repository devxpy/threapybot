import json
from multiprocessing.pool import ThreadPool

import pandas as pd
import requests_html as rh
from pyquery import PyQuery

ret = []


def main():
    urls = [
        "https://www.thebrightsessions.com/season-one",
        "https://www.thebrightsessions.com/season-two",
        "https://www.thebrightsessions.com/season-three",
        "https://www.thebrightsessions.com/season-four",
        "https://www.thebrightsessions.com/bonus-episodes",
        "https://www.thebrightsessions.com/season-six",
        "https://www.thebrightsessions.com/season-seven",
    ]
    for url in urls:
        s = rh.HTMLSession()
        r = s.get(url)
        r.raise_for_status()
        for el in r.html.find("a"):
            if "transcript" not in el.text.lower() or "transcripts" in el.text.lower():
                continue
            href = el.attrs["href"]
            if not href.startswith("http"):
                href = f"https://www.thebrightsessions.com{href}"
            pool.apply(get_transcripts, args=[href])


def get_transcripts(link):
    s = rh.HTMLSession()
    r = s.get(link)
    r.raise_for_status()

    h1 = r.html.pq.find("h1")[-1]
    title = h1.text

    try:
        h3 = r.html.pq.find("h3")[-1]
    except IndexError:
        h3 = h1
    else:
        title += "\n" + h3.text

    transcript = PyQuery(h3).parent().remove("h3").remove("h1").text()
    ret.append([link, title, transcript])
    with open("out.json", "w") as f:
        json.dump(ret, f, indent=2)


with ThreadPool(100) as pool:
    main()

with open("out.json") as f:
    ret = json.load(f)
    df = pd.DataFrame(ret, columns=["source", "title", "transcript"])
    df.sort_values(["source", "title"], inplace=True)
    df.to_csv("out.csv", index=False)
