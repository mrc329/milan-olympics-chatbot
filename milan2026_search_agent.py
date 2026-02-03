"""
milan2026_search_agent.py
─────────────────────────
Pulls Olympic storylines from RSS feeds only. No scraping, no external APIs.
All sources are structured feeds designed for programmatic access.

Sources
  1. Olympic Channel RSS   — IOC-owned podcast/story feed
       https://rss.art19.com/olympic-channel
  2. BBC Sport Olympics RSS — structured, reliable, updated during Games
       https://feeds.bbci.co.uk/sport/olympics/rss.xml
  3. Team USA RSS           — official US Olympic Committee news
       https://www.teamusa.com/news/rss
  4. AP News Olympics       — wire service (NBC/ESPN/all outlets pull from AP)
       https://apnews.com/hub/milan-cortina-2026-winter-olympics?format=rss

Olympic Content Filter
  RSS feeds cover ALL sports (NBA, NFL, soccer, etc). We filter to keep
  only chunks mentioning Milano Cortina 2026, Winter Olympics, or specific
  winter sports. Without this, Tyler/Sasha would talk about basketball.

Deduplication
  Before upserting, query Pinecone with the chunk's embedding.
  Score >= 0.92 -> already in index, skip.

Environment
  PINECONE_API_KEY   — secret (GitHub Actions)
  AGENT_LOG_LEVEL    — INFO | DEBUG
"""

import os
import sys
import re
import time
import hashlib
import logging
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Try to import feedparser for robust RSS parsing
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

# Import NLU module
try:
    from milan2026_nlu import analyze_content, enrich_metadata
    NLU_AVAILABLE = True
except ImportError:
    NLU_AVAILABLE = False

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
INDEX_NAME       = "milan-2026-olympics"
NAMESPACE        = "narratives"
DEDUP_THRESHOLD  = 0.92
MAX_WORDS        = 300
MIN_WORDS        = 40
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

RSS_FEEDS = [
    {
        "key":   "olympic_channel",
        "url":   "https://rss.art19.com/olympic-channel",
        "label": "Olympic Channel",
    },
    {
        "key":   "bbc_olympics",
        "url":   "https://feeds.bbci.co.uk/sport/olympics/rss.xml",
        "label": "BBC Sport Olympics",
    },
    {
        "key":   "team_usa",
        "url":   "https://www.teamusa.com/news/rss",
        "label": "Team USA",
    },
    {
        "key":   "ap_olympics",
        "url":   "https://apnews.com/hub/milan-cortina-2026-winter-olympics?format=rss",
        "label": "AP News Olympics",
    },
]

HEADERS = {
    "User-Agent": "MilanoCortina2026Bot/1.0 (storyline feed reader)"
}

# ═══════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════
log_level = os.getenv("AGENT_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("search_agent")

# Log feedparser availability
if FEEDPARSER_AVAILABLE:
    logger.info("feedparser available - robust RSS parsing enabled")
else:
    logger.warning("feedparser not available - using basic XML parser (may fail on malformed feeds)")

# Log NLU availability
if not NLU_AVAILABLE:
    logger.warning("NLU module not available - running without entity extraction")


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════
def deterministic_id(source_key: str, unique_field: str) -> str:
    """Stable 32-char hex ID so re-runs never create duplicates."""
    raw = f"{source_key}:{unique_field}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def truncate(text: str, max_words: int = MAX_WORDS) -> str:
    """Collapse whitespace and hard-cap word count."""
    words = text.split()
    return " ".join(words[:max_words]).strip()


def strip_html(text: str) -> str:
    """Remove any HTML tags that snuck into RSS descriptions."""
    return re.sub(r"<[^>]+>", " ", text)


# ═══════════════════════════════════════════════════════════
# RSS PARSING (with feedparser fallback)
# ═══════════════════════════════════════════════════════════
def parse_rss_feed(feed_def: dict) -> list[dict]:
    """
    Fetch one RSS 2.0 feed, return list of:
        { "id": str, "text": str, "source_key": str, "url": str, "title": str }

    Uses feedparser if available (handles malformed XML), falls back to
    stdlib xml.etree.ElementTree if feedparser not installed.
    """
    key   = feed_def["key"]
    url   = feed_def["url"]
    label = feed_def["label"]

    logger.info(f"Fetching RSS: {label} …")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"  RSS fetch failed ({label}): {e}")
        return []

    # Try feedparser first (handles malformed XML gracefully)
    if FEEDPARSER_AVAILABLE:
        try:
            feed = feedparser.parse(resp.content)
            
            # Check for parsing issues (bozo flag)
            if feed.bozo and feed.bozo_exception:
                logger.debug(f"  Feed had minor parsing issues (handled): {feed.bozo_exception}")
            
            chunks: list[dict] = []
            seen_titles: set[str] = set()

            for entry in feed.entries:
                title = entry.get('title', '').strip()
                
                # Try multiple fields for content
                desc = (
                    entry.get('description', '') or 
                    entry.get('summary', '') or 
                    (entry.get('content', [{}])[0].get('value', '') if entry.get('content') else '')
                ).strip()
                
                link = entry.get('link', '').strip()

                if not title and not desc:
                    continue

                # Dedupe within this feed
                if title in seen_titles:
                    continue
                seen_titles.add(title)

                body = strip_html(desc)
                text = truncate(f"{title}. {body}")

                if len(text.split()) < MIN_WORDS:
                    continue

                vec_id = deterministic_id(key, link or title)
                chunks.append({
                    "id":         vec_id,
                    "text":       text,
                    "source_key": key,
                    "url":        link,
                    "title":      title,
                })

            logger.info(f"  -> {len(chunks)} chunks from {label}")
            return chunks
            
        except Exception as e:
            logger.warning(f"  feedparser failed ({label}): {e}, trying XML parser")
            # Fall through to XML parser below

    # Fallback to basic XML parser
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        logger.warning(f"  RSS parse error ({label}): {e}")
        return []

    items = root.iter("item")
    chunks: list[dict] = []
    seen_titles: set[str] = set()

    for item in items:
        title_el   = item.find("title")
        desc_el    = item.find("description")
        link_el    = item.find("link")
        summary_el = item.find("summary")

        title   = (title_el.text   or "").strip() if title_el   is not None else ""
        desc    = (desc_el.text    or "").strip() if desc_el    is not None else ""
        summary = (summary_el.text or "").strip() if summary_el is not None else ""
        link    = (link_el.text    or "").strip() if link_el    is not None else ""

        body = desc or summary
        if not body and not title:
            continue

        # dedupe within this feed
        if title in seen_titles:
            continue
        seen_titles.add(title)

        body = strip_html(body)
        text = truncate(f"{title}. {body}")

        if len(text.split()) < MIN_WORDS:
            continue

        vec_id = deterministic_id(key, link or title)
        chunks.append({
            "id":         vec_id,
            "text":       text,
            "source_key": key,
            "url":        link,
            "title":      title,
        })

    logger.info(f"  -> {len(chunks)} chunks from {label}")
    return chunks


# ═══════════════════════════════════════════════════════════
# OLYMPIC CONTENT FILTER (NLU-lite)
# ═══════════════════════════════════════════════════════════
OLYMPIC_KEYWORDS = {
    # Event-specific (Milan 2026 only)
    "milano cortina", "milan cortina", "milano 2026", "milan 2026",
    "cortina 2026", "winter olympics 2026", "olympic winter games 2026",
    "2026 winter olympics", "2026 olympics",
    
    # Winter Olympics (generic but winter-specific)
    "winter olympics", "olympic winter games", "winter olympic",
    
    # Athletes & roles (catches "Olympic skater", "Olympic skier")
    "olympic skier", "olympic skater", "olympic snowboarder",
    "olympic athlete", "olympian",
    
    # Medal/competition terms
    "gold medal", "silver medal", "bronze medal", "olympic medal",
    "podium", "olympic champion", "flag bearer", "flagbearer",
    "opening ceremony", "closing ceremony",
    
    # Winter sports - general
    "alpine skiing", "alpine skier", "downhill ski", "slalom", "giant slalom",
    "figure skating", "figure skater", "ice skating", "ice skater",
    "ice hockey", "hockey player", "ice hockey arena",
    "speed skating", "speed skater", "speedskater", "short track",
    "bobsled", "bobsledder", "bobsleigh", "skeleton", "luge",
    "ski jumping", "ski jumper", "freestyle skiing", "freestyle skier",
    "cross-country ski", "nordic ski", "curling", "biathlon",
    "snowboard", "snowboarder", "nordic combined", "ski mountaineering",
    
    # Famous winter athletes (catches articles about them)
    "lindsey vonn", "mikaela shiffrin", "erin jackson", "brittany bowe",
    "ilia malinin", "nathan chen", "yuzuru hanyu",
    "chloe kim", "shaun white", "eileen gu",
    
    # Organizations
    "ioc", "international olympic committee",
    "usoc", "usopc", "team usa olympic",
}


def filter_olympic_content(chunks: list[dict]) -> list[dict]:
    """
    Keep only chunks mentioning Olympic-related keywords.
    
    Context-aware filtering:
    - AP Milano Cortina hub (ap_olympics): Keep ALL chunks (already filtered by AP)
    - Other feeds: Apply keyword filter to avoid NBA/NFL/soccer
    """
    if not chunks:
        return []
    
    logger.info(f"Filtering {len(chunks)} chunks for Olympic content …")
    kept = []
    
    for chunk in chunks:
        source_key = chunk.get("source_key", "")
        
        # AP Milano Cortina hub - trust it's already filtered
        if source_key == "ap_olympics":
            kept.append(chunk)
            continue
        
        # Other feeds - apply keyword filter
        text_lower = chunk["text"].lower()
        if any(kw in text_lower for kw in OLYMPIC_KEYWORDS):
            kept.append(chunk)
        else:
            logger.debug(f"  FILTERED (no Olympic keywords): {chunk['url'] or chunk['source_key']}")
    
    logger.info(f"  {len(kept)} Olympic-related / {len(chunks) - len(kept)} filtered out")
    return kept


# ═══════════════════════════════════════════════════════════
# DEDUPLICATE AGAINST PINECONE
# ═══════════════════════════════════════════════════════════
def deduplicate(chunks: list[dict], model: SentenceTransformer, index) -> list[dict]:
    """
    Embed each chunk, query Pinecone top-1 in narratives namespace.
    Drop anything scoring >= DEDUP_THRESHOLD.
    """
    if not chunks:
        return []

    logger.info(f"Deduplicating {len(chunks)} chunks …")
    new_chunks = []

    for chunk in chunks:
        vec = model.encode(chunk["text"]).tolist()
        try:
            result = index.query(vector=vec, top_k=1, namespace=NAMESPACE, include_metadata=False)
            matches = result.get("matches", [])
            if matches and matches[0].get("score", 0) >= DEDUP_THRESHOLD:
                logger.debug(f"  SKIP (sim={matches[0]['score']:.3f}): {chunk['url'] or chunk['source_key']}")
                continue
        except Exception as e:
            logger.warning(f"  Dedup query error: {e} — keeping chunk (fail-open)")

        new_chunks.append(chunk)

    logger.info(f"  {len(new_chunks)} new / {len(chunks) - len(new_chunks)} dupes")
    return new_chunks


# ═══════════════════════════════════════════════════════════
# EMBED + UPSERT
# ═══════════════════════════════════════════════════════════
def upsert_chunks(chunks: list[dict], model: SentenceTransformer, index):
    """Batch-embed and upsert into the narratives namespace with NLU enrichment."""
    if not chunks:
        logger.info("Nothing to upsert.")
        return

    logger.info(f"Embedding + upserting {len(chunks)} chunks …")
    texts   = [c["text"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=False).tolist()

    records = []
    nlu_enriched = 0
    
    for chunk, vec in zip(chunks, vectors):
        metadata = {
            "text":        chunk["text"],
            "source_key":  chunk["source_key"],
            "url":         chunk["url"],
            "doc_type":    "narrative",
            "fetched_at":  datetime.now(timezone.utc).isoformat(),
        }
        
        # Add NLU enrichment if available
        if NLU_AVAILABLE:
            try:
                nlu_result = analyze_content(
                    text=chunk["text"],
                    title=chunk.get("title", ""),
                    min_quality=0.2  # Low threshold for RSS (we already filtered by keywords)
                )
                if nlu_result:
                    nlu_meta = enrich_metadata(nlu_result)
                    metadata.update(nlu_meta)
                    nlu_enriched += 1
                    logger.debug(f"NLU enriched: {len(nlu_result.athletes)} athletes, {len(nlu_result.topics)} topics")
            except Exception as e:
                logger.warning(f"NLU enrichment failed for chunk {chunk['id']}: {e}")
        
        records.append({
            "id": chunk["id"],
            "values": vec,
            "metadata": metadata
        })

    index.upsert(vectors=records, namespace=NAMESPACE)
    logger.info(f"Upsert complete. NLU enriched: {nlu_enriched}/{len(chunks)} chunks")


# ═══════════════════════════════════════════════════════════
# SUMMARY REPORTING
# ═══════════════════════════════════════════════════════════
def log_feed_summary(all_chunks: list[dict], olympic_chunks: list[dict]):
    """
    Log a summary showing what was captured from each feed.
    Helps understand which feeds are working and what's being filtered.
    """
    from collections import Counter
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("FEED SUMMARY")
    logger.info("=" * 70)
    
    # Group by source
    raw_counts = Counter(c["source_key"] for c in all_chunks)
    filtered_counts = Counter(c["source_key"] for c in olympic_chunks)
    
    # Get feed labels
    feed_labels = {f["key"]: f["label"] for f in RSS_FEEDS}
    
    for feed_key in sorted(raw_counts.keys()):
        raw = raw_counts[feed_key]
        filtered = filtered_counts.get(feed_key, 0)
        removed = raw - filtered
        label = feed_labels.get(feed_key, feed_key)
        
        logger.info(f"  {label:25} │ Raw: {raw:3} │ Kept: {filtered:3} │ Filtered: {removed:3}")
    
    logger.info("=" * 70)
    logger.info(f"  TOTALS: Raw: {len(all_chunks)}, Kept: {len(olympic_chunks)}, Filtered: {len(all_chunks) - len(olympic_chunks)}")
    logger.info("=" * 70)
    logger.info("")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    t0 = time.time()

    api_key = os.getenv("PINECONE_API_KEY", "")
    if not api_key:
        logger.error("PINECONE_API_KEY not set.")
        sys.exit(1)

    pc    = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)
    logger.info(f"Connected -> {INDEX_NAME}")

    logger.info(f"Loading {EMBEDDING_MODEL} …")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Model ready.")

    # 1. Fetch RSS feeds
    all_chunks: list[dict] = []
    for feed in RSS_FEEDS:
        all_chunks.extend(parse_rss_feed(feed))

    logger.info(f"Total raw chunks: {len(all_chunks)}")

    # 2. Filter for Olympic content
    olympic_chunks = filter_olympic_content(all_chunks)
    
    # 2.5. Log feed summary
    log_feed_summary(all_chunks, olympic_chunks)

    # 3. Deduplicate against Pinecone
    new_chunks = deduplicate(olympic_chunks, model, index)

    # 4. Upsert
    upsert_chunks(new_chunks, model, index)

    logger.info(f"Search agent done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
