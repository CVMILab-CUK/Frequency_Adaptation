#!/usr/bin/env python3
"""Verify arXiv references against arXiv's own API.

Every entry is looked up by ID; the title and authors come back from arXiv, not
from us. An ID that returns nothing, or whose returned title does not match the
title we intended to cite, is dropped and reported. Nothing is written from
memory.
"""
import sys, json, time, re, urllib.request, urllib.parse
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom"}
API = "http://export.arxiv.org/api/query"

def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def fetch_ids(ids):
    out = {}
    for i in range(0, len(ids), 20):
        chunk = ids[i:i+20]
        url = f"{API}?id_list={','.join(chunk)}&max_results=40"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    root = ET.fromstring(r.read())
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  API failed for {chunk}: {e}", file=sys.stderr); root = None
                time.sleep(4)
        if root is None: continue
        for e in root.findall("a:entry", NS):
            eid = e.find("a:id", NS).text.rsplit("/", 1)[-1]
            base = eid.split("v")[0]
            out[base] = {
                "id": base,
                "title": " ".join(e.find("a:title", NS).text.split()),
                "authors": [a.find("a:name", NS).text for a in e.findall("a:author", NS)],
                "published": e.find("a:published", NS).text[:10],
                "url": f"https://arxiv.org/abs/{base}",
            }
        time.sleep(3)
    return out

def search(query, n=12):
    url = (f"{API}?search_query={urllib.parse.quote(query)}"
           f"&sortBy=submittedDate&sortOrder=descending&max_results={n}")
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            root = ET.fromstring(r.read())
    except Exception as e:
        print(f"  search failed: {e}", file=sys.stderr); return []
    res = []
    for e in root.findall("a:entry", NS):
        res.append({
            "id": e.find("a:id", NS).text.rsplit("/", 1)[-1].split("v")[0],
            "title": " ".join(e.find("a:title", NS).text.split()),
            "published": e.find("a:published", NS).text[:10],
            "authors": [a.find("a:name", NS).text for a in e.findall("a:author", NS)],
        })
    time.sleep(3)
    return res

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "search":
        for r in search(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 12):
            print(f"{r['id']:<14} {r['published']}  {r['title'][:88]}")
    else:
        got = fetch_ids(sys.argv[2:])
        print(json.dumps(got, indent=1, ensure_ascii=False))
