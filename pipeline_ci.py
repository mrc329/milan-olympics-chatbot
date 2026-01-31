"""
pipeline_ci.py — GitHub Actions runner for Milan 2026 live vector updates.
===========================================================================
Runs every 30 minutes via .github/workflows/update_vectors.yml.

Two types of content are fetched each run:

  1. EVENT RESULTS — sport-specific pages with medal tables.
     Wikipedia page pattern: "{Sport} at the 2026 Winter Olympics – {Event}"
     Extraction: parse HTML tables → top-3 finishers.
     Vector ID:  result_{sport}_{event}_{date}

  2. NARRATIVE PAGES — ceremonies, controversies, the main Games page.
     Wikipedia pages: "2026 Winter Olympics opening ceremony", etc.
     Extraction: strip HTML → plain text → truncate to 800 words.
     Vector ID:  narrative_{page_key}  (static, so each run overwrites
                 with the latest Wikipedia text automatically)

Design constraints:
  - No persistent state between runs (fresh container each time)
  - Must be fast — model loads from Actions cache, script finishes < 30s
  - Must be idempotent — deterministic vector IDs, upsert = insert or overwrite
  - Must be silent outside the Games — exits immediately if not Feb 6–22
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import os
import re
import sys
import time
import logging
from datetime import datetime
from html.parser import HTMLParser

import pandas as pd
import requests
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout
)
logger = logging.getLogger("milan2026_ci")


# ─────────────────────────────────────────────
# DATE GUARD — exit fast outside the Games
# ─────────────────────────────────────────────
now = datetime.utcnow()
GAMES_START = datetime(2026, 2, 6)
GAMES_END   = datetime(2026, 2, 22, 23, 59)

if now < GAMES_START or now > GAMES_END:
    logger.info(f"Outside Games window ({now.strftime('%b %d')}). Exiting.")
    sys.exit(0)

logger.info(f"Games in progress. Running update ({now.strftime('%b %d %H:%M')} UTC)")


# ─────────────────────────────────────────────
# PINECONE
# ─────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    logger.error("PINECONE_API_KEY not set. Check GitHub Secrets.")
    sys.exit(1)

pc    = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("milan-2026-olympics")
logger.info(f"Pinecone connected. Vectors: {index.describe_index_stats()['total_vector_count']}")


# ─────────────────────────────────────────────
# EMBEDDING MODEL
# ─────────────────────────────────────────────
logger.info("Loading embedding model…")
model = SentenceTransformer("all-MiniLM-L6-v2")   # 384-dim, cached by Actions
logger.info("Model ready.")


def embed(text: str) -> list:
    return model.encode(
        str(text).replace("\n", " ").strip(),
        convert_to_tensor=False,
        show_progress_bar=False
    ).tolist()


# ─────────────────────────────────────────────
# WIKIPEDIA — shared fetch (both paths use this)
# ─────────────────────────────────────────────
def fetch_wiki_html(page_title: str) -> str | None:
    """Fetch any Wikipedia page via MediaWiki API. Returns raw HTML or None."""
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "parse", "page": page_title, "prop": "text", "format": "json"},
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return None
        return data.get("parse", {}).get("text", {}).get("*")
    except Exception as e:
        logger.warning(f"Fetch failed for '{page_title}': {e}")
        return None


# ─────────────────────────────────────────────
# PATH 1 — EVENT RESULTS (medal tables)
# ─────────────────────────────────────────────
def extract_medalists(html: str) -> list[dict] | None:
    """Parse top-3 finishers from a Wikipedia results table."""
    try:
        tables = pd.read_html(html)
    except Exception:
        return None

    for tbl in tables:
        cols_lower = [str(c).lower() for c in tbl.columns]

        has_rank  = any(k in cols_lower for k in ["rank", "#", "place"])
        has_name  = any(k in cols_lower for k in ["athlete", "name", "skater", "team", "player"])
        has_medal = any(k in cols_lower for k in ["medal", "gold"])

        if not (has_rank or has_medal) or not has_name:
            continue

        col_map = {}
        for c in tbl.columns:
            cl = str(c).lower()
            if   cl in ("rank", "#", "place"):                             col_map[c] = "rank"
            elif cl in ("athlete", "name", "skater", "team", "player"):    col_map[c] = "name"
            elif cl in ("noc", "country", "nation"):                       col_map[c] = "country"
            elif cl in ("time", "score", "points", "total", "result"):     col_map[c] = "score"

        tbl = tbl.rename(columns=col_map)

        results = []
        for i, row in tbl.head(3).iterrows():
            results.append({
                "rank":    i + 1,
                "name":    str(row.get("name",    "Unknown")).strip(),
                "country": str(row.get("country", "")).strip(),
                "score":   str(row.get("score",   "")).strip()
            })

        if results:
            return results

    return None


def slug(s: str) -> str:
    return s.lower().replace(" ", "_").replace("'", "").replace("–", "").replace("-", "_")


MEDAL_LABELS = {1: "Gold", 2: "Silver", 3: "Bronze"}

def upsert_result(sport: str, event: str, date_str: str, medalists: list[dict]):
    """Build text block, embed, upsert with deterministic ID."""
    lines = []
    for m in medalists:
        label = MEDAL_LABELS.get(m["rank"], f"#{m['rank']}")
        line  = f"  {label}: {m['name']} ({m['country']})"
        if m["score"]:
            line += f" — {m['score']}"
        lines.append(line)

    text = (
        f"Milan 2026 Olympics — Event Result\n"
        f"Sport: {sport}\n"
        f"Event: {event}\n"
        f"Date: {date_str}\n"
        f"Medalists:\n" + "\n".join(lines)
    )

    vector_id = f"result_{slug(sport)}_{slug(event)}_{slug(date_str)}"

    index.upsert(vectors=[{
        "id":     vector_id,
        "values": embed(text),
        "metadata": {
            "doc_type": "event_result",
            "sport":    sport,
            "event":    event,
            "date":     date_str,
            "gold":     medalists[0]["name"] if len(medalists) > 0 else "",
            "silver":   medalists[1]["name"] if len(medalists) > 1 else "",
            "bronze":   medalists[2]["name"] if len(medalists) > 2 else "",
            "text":     text
        }
    }])
    logger.info(f"✅ Upserted result: {vector_id}")


# ─────────────────────────────────────────────
# PATH 2 — NARRATIVE PAGES (prose: ceremonies, controversies, etc.)
# ─────────────────────────────────────────────
class _TagStripper(HTMLParser):
    """Minimal HTML → plain text (no external deps needed)."""
    def __init__(self):
        super().__init__()
        self._chunks: list[str] = []
    def handle_data(self, data):
        self._chunks.append(data)
    def get_text(self) -> str:
        return "".join(self._chunks)


def html_to_text(html: str) -> str:
    """Strip tags, collapse whitespace."""
    s = _TagStripper()
    s.feed(html)
    text = s.get_text()
    text = re.sub(r'\n\s*\n+', '\n\n', text)   # multiple blank lines → one
    text = re.sub(r'[ \t]+', ' ', text)         # tab/space runs → single space
    return text.strip()


NARRATIVE_MAX_WORDS = 800

def extract_narrative(html: str) -> str:
    """Convert HTML to plain text, truncated to NARRATIVE_MAX_WORDS."""
    text  = html_to_text(html)
    words = text.split()
    if len(words) > NARRATIVE_MAX_WORDS:
        text = " ".join(words[:NARRATIVE_MAX_WORDS])
    return text


def upsert_narrative(vector_id: str, doc_type: str, title: str, text: str):
    """Embed and upsert a prose vector. Static ID = auto-updates each run."""
    index.upsert(vectors=[{
        "id":     vector_id,
        "values": embed(text),
        "metadata": {
            "doc_type": doc_type,
            "title":    title,
            "text":     text
        }
    }])
    logger.info(f"✅ Upserted narrative: {vector_id}")


# ─────────────────────────────────────────────
# DATA — NARRATIVE PAGES
# Wikipedia creates dedicated pages for these at every Olympics.
# Vector IDs are static so each run just overwrites with current text.
#
# (wikipedia_page_title, vector_id, doc_type, friendly_title)
# ─────────────────────────────────────────────
NARRATIVE_PAGES = [
    ("2026 Winter Olympics opening ceremony",
     "narrative_opening_ceremony",
     "ceremony",
     "Milan 2026 — Opening Ceremony"),

    ("2026 Winter Olympics closing ceremony",
     "narrative_closing_ceremony",
     "ceremony",
     "Milan 2026 — Closing Ceremony"),

    ("2026 Winter Olympics controversies",
     "narrative_controversies",
     "controversy",
     "Milan 2026 — Controversies"),

    # Master Games page — running narrative, venues, politics, notable moments
    ("2026 Winter Olympics",
     "narrative_main_games_page",
     "overview",
     "Milan 2026 — Games Overview"),
]


# ─────────────────────────────────────────────
# DATA — MEDAL-TABLE EVENTS
# (sport, event, date)  — date is approximate, used in vector ID
# ─────────────────────────────────────────────
PRIORITY_EVENTS = [
    # Figure Skating
    ("Figure skating",            "Men's singles",                  "Feb 13"),
    ("Figure skating",            "Women's singles",                "Feb 15"),
    ("Figure skating",            "Pairs",                          "Feb 11"),
    ("Figure skating",            "Ice dance",                      "Feb 17"),

    # Ice Hockey
    ("Ice hockey",                "Men's tournament",               "Feb 22"),
    ("Ice hockey",                "Women's tournament",             "Feb 20"),

    # Alpine Skiing
    ("Alpine skiing",             "Men's downhill",                 "Feb 7"),
    ("Alpine skiing",             "Women's downhill",               "Feb 8"),
    ("Alpine skiing",             "Men's slalom",                   "Feb 16"),
    ("Alpine skiing",             "Women's slalom",                 "Feb 14"),

    # Curling
    ("Curling",                   "Men's singles",                  "Feb 22"),
    ("Curling",                   "Women's singles",                "Feb 20"),
    ("Curling",                   "Mixed doubles",                  "Feb 8"),

    # Speed Skating
    ("Speed skating",             "Men's 500 metres",               "Feb 11"),
    ("Speed skating",             "Women's 500 metres",             "Feb 13"),
    ("Speed skating",             "Men's 1000 metres",              "Feb 14"),

    # Biathlon
    ("Biathlon",                  "Men's sprint",                   "Feb 7"),
    ("Biathlon",                  "Women's sprint",                 "Feb 8"),
    ("Biathlon",                  "Men's individual",               "Feb 10"),
    ("Biathlon",                  "Women's individual",             "Feb 11"),

    # Cross-Country
    ("Cross-country skiing",      "Men's skiathlon",                "Feb 6"),
    ("Cross-country skiing",      "Women's skiathlon",              "Feb 6"),

    # Freestyle Skiing
    ("Freestyle skiing",          "Men's moguls",                   "Feb 8"),
    ("Freestyle skiing",          "Women's moguls",                 "Feb 6"),

    # Short Track
    ("Short track speed skating", "Men's 1000 metres",              "Feb 7"),
    ("Short track speed skating", "Women's 500 metres",             "Feb 9"),

    # Ski Jumping
    ("Ski jumping",               "Men's normal hill individual",   "Feb 6"),
    ("Ski jumping",               "Women's normal hill individual", "Feb 8"),
]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    uploaded = 0
    not_yet  = 0

    # ── 1. Narrative pages ──────────────────────────────────────────
    # Fetched every run. Static vector IDs mean Wikipedia edits
    # automatically flow into the index on the next cycle.
    for page_title, vector_id, doc_type, friendly_title in NARRATIVE_PAGES:
        html = fetch_wiki_html(page_title)
        if html is None:
            not_yet += 1
            logger.info(f"⏳ No page yet: {friendly_title}")
        else:
            text = extract_narrative(html)
            upsert_narrative(vector_id, doc_type, friendly_title, text)
            uploaded += 1
        time.sleep(0.3)

    # ── 2. Medal-table events ───────────────────────────────────────
    # Page pattern: "{Sport} at the 2026 Winter Olympics – {Event}"
    for sport, event, date_str in PRIORITY_EVENTS:
        page_title = f"{sport} at the 2026 Winter Olympics – {event}"
        html = fetch_wiki_html(page_title)

        if html is None:
            not_yet += 1
            logger.info(f"⏳ No page yet: {sport} — {event}")
            time.sleep(0.3)
            continue

        medalists = extract_medalists(html)
        if medalists is None:
            not_yet += 1
            logger.info(f"⏳ Page exists, no results yet: {sport} — {event}")
            time.sleep(0.3)
            continue

        upsert_result(sport, event, date_str, medalists)
        uploaded += 1
        time.sleep(0.3)

    # ── summary ─────────────────────────────────────────────────────
    final_count = index.describe_index_stats()["total_vector_count"]
    logger.info("")
    logger.info("━━━ Run complete ━━━")
    logger.info(f"  Upserted:  {uploaded}")
    logger.info(f"  Not yet:   {not_yet}")
    logger.info(f"  Total vectors in index: {final_count}")


if __name__ == "__main__":
    main()
