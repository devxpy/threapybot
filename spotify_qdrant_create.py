import json
import os

import requests_html
import torch
from qdrant_client import models, QdrantClient
from qdrant_client.models import VectorParams
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "spotify-genres"


def main():
    encoder = SentenceTransformer(
        "thenlper/gte-large", device="mps" if torch.has_mps else "cuda"
    )  # thenlper/gte-base

    documents = scrape_playlists()
    # with open("genres.json") as f:
    #     documents = json.load(f)

    # Create a client object for Qdrant.
    qdrant = QdrantClient(
        "https://303757f4-9127-4df0-a9e8-9669f997f742.us-east4-0.gcp.cloud.qdrant.io:6333",
        api_key=os.environ["QDRANT_API_KEY"],
    )

    # Related vectors need to be added to a collection. Create a new collection for your startup vectors.
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
            distance=models.Distance.COSINE,
        ),
    )

    qdrant.upload_records(
        collection_name=COLLECTION_NAME,
        records=[
            models.Record(
                id=idx, vector=encoder.encode(doc["description"]).tolist(), payload=doc
            )
            for idx, doc in enumerate(documents)
        ],
    )

    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=encoder.encode("depressed").tolist(),
        limit=9,
    )
    for hit in hits:
        print(hit.payload, "score:", hit.score)


def scrape_playlists():
    documents = []
    s = requests_html.HTMLSession()

    r = s.get("https://everynoise.com/genrewords.html")
    r.raise_for_status()
    genrewords = {}
    for el in r.html.find("tr"):
        genre_slug = get_genre_slug(el.find("td")[0].text)
        description = ", ".join(el.find("td")[1].text.split(" "))
        genrewords[genre_slug] = description

    r = s.get("https://everynoise.com/everynoise1d.cgi?scope=all")
    r.raise_for_status()
    for el in r.html.find("tr"):
        playlist_url = el.find("td")[1].find("a", first=True).attrs["href"]
        description = el.find("td")[2].find("a", first=True).text.strip()
        genre_name = description
        genre_slug = get_genre_slug(description)
        try:
            description += ", " + genrewords[genre_slug]
        except KeyError:
            pass
        documents.append(
            {
                "playlist_url": playlist_url,
                "genre_name": genre_name,
                "genre_slug": genre_slug,
                "description": description,
            }
        )
    with open("genres.json", "w") as f:
        json.dump(documents, f, indent=2)
    return documents


def get_genre_slug(text):
    return "".join(a for a in text.lower() if a.isalnum())


if __name__ == "__main__":
    main()
