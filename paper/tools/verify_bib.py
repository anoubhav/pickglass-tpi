"""Independent re-verification of refs.bib: DOI -> Crossref (fallback DataCite / doi.org CSL),
arXiv eprint -> arXiv API, URL -> HTTP GET. Writes verify_report.json + prints a table."""
import json, re, sys, time, difflib, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from pybtex.database import parse_file
from pybtex.textutils import abbreviate
BIB = sys.argv[1]
OUT = sys.argv[2]
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}

def get(url, accept=None, timeout=30):
    h = dict(UA)
    if accept: h["Accept"] = accept
    req = urllib.request.Request(url, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.geturl(), r.read()
    except urllib.error.HTTPError as e:
        return e.code, url, e.read() if hasattr(e, "read") else b""
    except Exception as e:
        return -1, url, str(e).encode()

def norm(s):
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[`'\"^~=.uvHckb]\{?([a-zA-Z])\}?", r"\1", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()

def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()

db = parse_file(BIB)
rows = []
for key, e in db.entries.items():
    f = {k.lower(): v for k, v in e.fields.items()}
    title = f.get("title", "")
    year = f.get("year", "")
    authors = [p.last_names[0] if p.last_names else str(p) for p in e.persons.get("author", [])]
    rec = {"key": key, "type": e.type, "title": title, "year": year, "first_author": authors[0] if authors else ""}
    doi = f.get("doi")
    if doi:
        st, _, body = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
        src = "crossref"
        meta = None
        if st == 200:
            m = json.loads(body)["message"]
            ttl = (m.get("title") or [""])[0]
            yr = None
            for k in ("published-print", "issued", "published-online"):
                if m.get(k, {}).get("date-parts"):
                    yr = m[k]["date-parts"][0][0]; break
            au = [a.get("family", a.get("name", "")) for a in m.get("author", [])]
            meta = {"title": ttl, "year": yr, "authors": au[:3], "container": (m.get("container-title") or [""])[0],
                    "volume": m.get("volume"), "issue": m.get("issue"), "page": m.get("page")}
        else:
            st2, _, body2 = get("https://api.datacite.org/dois/" + urllib.parse.quote(doi))
            src = "datacite"
            if st2 == 200:
                a = json.loads(body2)["data"]["attributes"]
                meta = {"title": a["titles"][0]["title"], "year": a.get("publicationYear"),
                        "authors": [c.get("familyName", c.get("name")) for c in a.get("creators", [])][:3],
                        "container": a.get("publisher")}
            st = st2
        rec.update({"check": "doi", "src": src, "status": st, "meta": meta})
        if meta:
            rec["title_sim"] = round(sim(title, meta["title"]), 3)
            rec["year_ok"] = str(meta.get("year")) == str(year)
            rec["author_ok"] = (not authors) or any(norm(authors[0]) and norm(authors[0]).split()[-1] in norm(x or "") for x in meta["authors"]) or not meta["authors"]
            for fld in ("volume", "number", "pages"):
                if fld in f and meta.get({"number": "issue", "pages": "page"}.get(fld, fld)):
                    mv = str(meta[{"number": "issue", "pages": "page"}.get(fld, fld)]).replace("--", "-")
                    bv = f[fld].replace("--", "-")
                    rec[fld + "_ok"] = (mv == bv) or (fld == "pages" and mv.split("-")[0] == bv.split("-")[0])
        time.sleep(0.2)
    elif f.get("eprint"):
        aid = f["eprint"]
        st, _, body = get("http://export.arxiv.org/api/query?id_list=" + aid)
        meta = None
        if st == 200:
            ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
            root = ET.fromstring(body)
            ent = root.find("a:entry", ns)
            if ent is not None and ent.find("a:title", ns) is not None:
                meta = {"title": " ".join(ent.find("a:title", ns).text.split()),
                        "published": ent.find("a:published", ns).text[:10],
                        "authors": [a.find("a:name", ns).text for a in ent.findall("a:author", ns)][:8],
                        "comment": (ent.find("x:comment", ns).text if ent.find("x:comment", ns) is not None else None),
                        "journal_ref": (ent.find("x:journal_ref", ns).text if ent.find("x:journal_ref", ns) is not None else None)}
        rec.update({"check": "arxiv", "status": st, "meta": meta})
        if meta:
            rec["title_sim"] = round(sim(title, meta["title"]), 3)
            rec["year_ok"] = meta["published"][:4] == year
            rec["author_ok"] = (not authors) or any(norm(authors[0]).split()[-1] in norm(x) for x in meta["authors"])
        time.sleep(3.1)
    else:
        url = f.get("url") or re.sub(r"\\url\{(.*)\}", r"\1", f.get("howpublished", ""))
        if url.startswith("http"):
            st, final, body = get(url)
            txt = body.decode("utf-8", "ignore") if isinstance(body, bytes) else ""
            m = re.search(r"<title[^>]*>(.*?)</title>", txt, re.S | re.I)
            rec.update({"check": "url", "url": url, "status": st, "final": final, "page_title": (m.group(1).strip()[:150] if m else None), "bytes": len(body)})
        else:
            rec.update({"check": "none"})
    rows.append(rec)
    flag = ""
    if rec.get("title_sim") is not None and rec["title_sim"] < 0.9: flag += " TITLE?"
    if rec.get("year_ok") is False: flag += " YEAR?"
    if rec.get("author_ok") is False: flag += " AUTHOR?"
    for fld in ("volume", "number", "pages"):
        if rec.get(fld + "_ok") is False: flag += f" {fld.upper()}?"
    print(f"{key:32s} {rec['check']:6s} {rec.get('status')} sim={rec.get('title_sim')}{flag}", flush=True)
json.dump(rows, open(OUT, "w"), indent=1, default=str)
