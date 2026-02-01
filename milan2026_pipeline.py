"""
milan2026_pipeline.py
=====================
Winter Olympics-focused pipeline for Milan 2026.

Flows (in execution order):
  1. Narratives    — ceremony / cultural pages.  Always runs in PRE + LIVE.
  2. Rumors        — unconfirmed reports (performer lineups, schedule changes).
                     PRE + LIVE.  Each rumor carries a confidence level and a
                     source.  A rumor can be promoted to a narrative if it gets
                     confirmed on a subsequent run.
  3. Injuries      — injury / fitness status for key athletes.  PRE + LIVE.
                     Each injury record has a severity (low / moderate / high)
                     and affects the athlete's vector on the next athlete pass.
  4. Events        — Winter event results.  LIVE only.
  5. Athletes      — enriched profiles.  Always runs.  Cross-references
                     EVENT_RESULTS, INJURIES, and RUMORS to build a single
                     rich vector per athlete.
  6. Upset detect         — LIVE only, after events.  Any individual
                           medalist not on the favorites roster gets a
                           dedicated upset vector.  Skips team events.
  7. Country upset detect — LIVE only, after events.  Three signals:
                           • team_event: favored country lost gold in a
                             team event (hockey, curling).
                           • surge: a country's gold count exceeds its
                             historical baseline by more than the threshold.
                           • shutout: an expected country is entirely absent
                             from a podium.

Freshness SLAs (target max staleness during LIVE):
  narrative       60 min
  rumor           20 min   ← rumors confirm or die fast
  injury          15 min   ← can flip same-day
  athlete         30 min
  event           15 min
  upset            5 min
  country_upset    5 min   ← same urgency as individual upsets

Tracks every vector touched this run and prints a GitHub-Actions-friendly
summary at the end.
"""

from datetime import datetime, timezone
import logging
import time
import re

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
# Format: timestamp (UTC, seconds), level, message.
# GitHub Actions captures stdout at all levels; local dev can set
# PIPELINE_LOG_LEVEL=DEBUG to see fetch noise.
import os as _os
logging.basicConfig(
    level=getattr(logging, _os.getenv("PIPELINE_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    force=True,          # override any root config already set
)
log = logging.getLogger("milan2026_pipeline")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GAMES_START = datetime(2026, 2, 5, tzinfo=timezone.utc)   # women's hockey prelims start Feb 5, day before Opening Ceremony
GAMES_END   = datetime(2026, 2, 22, 23, 59, tzinfo=timezone.utc)

FRESHNESS_SLA_MINUTES = {
    "narrative":      60,
    "rumor":          20,   # rumors confirm or die fast — poll often
    "injury":         15,   # injury status can flip same-day; same cadence as events
    "athlete":        30,
    "event":          15,
    "upset":           5,
    "country_upset":   5,   # same urgency as individual upsets
}

# Known favorites / defending champions.
# Upset detection checks gold medalists against this list.
# IMPORTANT: slugs here must match what slug() produces.  Characters like
# ø and á get stripped by the regex, so "Bjørgen" → "bj_rgen" and
# "Ledecká" → "ledeck".  Verify with: print(slug("Name"))
KNOWN_FAVORITES = {
    "yuzuru_hanyu",
    "ester_ledeck",           # Ester Ledecká — á stripped
    "jessie_diggins",
    "john_shuster",
    "kendall_coyne_schofield",
    "danny_o_shea",
    "lee_stecklein",
    "mikaela_shiffrin",
    "irene_schouten",         # defending Olympic champion
    "therese_johaug",
}

# ── Team events ──────────────────────────────────────────────
# Individual upset detection skips these (team names like "USA Women"
# can't match an individual-athlete slug).  Country-level detection
# handles them instead, using TEAM_EVENT_FAVORITES below.
TEAM_EVENTS = {
    "Women's ice hockey tournament",
    "Men's curling",
}

# ── Country-level upset config ───────────────────────────────
# TEAM_EVENT_FAVORITES: which country is expected to win gold in each
# team event.  If someone else wins gold → country_upset vector.
TEAM_EVENT_FAVORITES = {
    "Women's ice hockey tournament": "USA",   # USA women historically dominant
    "Men's curling":                 "USA",   # Shuster's team defending 2018 gold
}

# HISTORICAL_GOLD_BASELINE: how many golds each country is realistically
# expected to win across the full set of events we're tracking.  Used by
# surge detection.  Countries not listed default to 0.
HISTORICAL_GOLD_BASELINE = {
    "USA": 1,
    "NOR": 1,   # cross-country powerhouse
    "JPN": 1,   # figure skating
    "SWE": 0,
    "NED": 0,   # speed skating specialist
    "CAN": 0,
    "CZE": 0,
    "FIN": 0,
}

# COUNTRY_SURGE_THRESHOLD: a country must exceed its gold baseline by
# MORE than this number to trigger a surge vector.  At 1, a country
# that picks up exactly one extra gold is "nice run" not "surge".
# Two extra golds is a genuine surprise worth surfacing.
COUNTRY_SURGE_THRESHOLD = 1

# EVENT_EXPECTED_COUNTRIES: for shutout detection.  Which countries are
# expected to appear somewhere on the podium for each event.  If an
# expected country is entirely absent → shutout vector.
EVENT_EXPECTED_COUNTRIES = {
    "Women's downhill alpine skiing":      {"USA", "CZE"},
    "Men's figure skating free skate":     {"JPN"},            # JPN sweep in stubs
    "Women's ice hockey tournament":       {"USA", "CAN"},
    "Women's cross-country skiathlon":     {"NOR", "USA"},
    "Men's curling":                       {"USA"},
    "Women's 500m speed skating":          {"USA", "NED"},
}

# ─────────────────────────────────────────────
# PINECONE + EMBEDDING MODEL
# ─────────────────────────────────────────────
# Only initialised when PINECONE_API_KEY is present (CI / production).
# When it's absent (tests, local dev) everything falls back to the
# in-memory VECTOR_STORE below — no embedding, no network call.
INDEX_NAME   = "milan-2026-olympics"
MODEL_NAME   = "all-MiniLM-L6-v2"   # must match the app's query model

_pinecone_index = None
_embedder       = None

def _init_pinecone():
    """Connect to Pinecone and load the embedding model.
    Called once at the top of main() when the API key is available."""
    global _pinecone_index, _embedder
    from pinecone import Pinecone
    from sentence_transformers import SentenceTransformer

    log.info("connecting to Pinecone index '%s'…", INDEX_NAME)
    pc = Pinecone(api_key=_os.getenv("PINECONE_API_KEY"))
    _pinecone_index = pc.Index(INDEX_NAME)
    stats = _pinecone_index.describe_index_stats()
    log.info("Pinecone ready — %d vectors currently in index", stats["total_vector_count"])

    log.info("loading embedding model '%s'…", MODEL_NAME)
    _embedder = SentenceTransformer(MODEL_NAME)
    log.info("embedding model ready (dim=%d)", _embedder.get_sentence_embedding_dimension())

# ─────────────────────────────────────────────
# VECTOR STORE (in-memory fallback for tests)
# ─────────────────────────────────────────────
VECTOR_STORE = {}   # vector_id → {text, metadata}

def upsert_vector(vector_id: str, text: str, metadata: dict) -> str:
    """Upsert a vector.

    If Pinecone is initialised (production / CI):
        embed text → upsert to Pinecone → mirror to VECTOR_STORE.
    Otherwise (tests / local dev):
        write to VECTOR_STORE only.
    """
    action = "inserted" if vector_id not in VECTOR_STORE else "updated"

    if _pinecone_index is not None:
        # Real path: embed + upsert
        embedding = _embedder.encode(text).tolist()
        metadata_with_text = {**metadata, "text": text}
        _pinecone_index.upsert(vectors=[{
            "id":       vector_id,
            "values":   embedding,
            "metadata": metadata_with_text,
        }])

    # Always mirror to in-memory store (tests read it for assertions)
    VECTOR_STORE[vector_id] = {"text": text, "metadata": metadata}
    log.info("upsert %-8s → %s", action.upper(), vector_id)
    return action

# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────
def resolve_mode(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    if now < GAMES_START:
        return "PRE_GAMES"
    if GAMES_START <= now <= GAMES_END:
        return "LIVE_GAMES"
    return "DORMANT"

def freshness_metadata(source: str, volatility: str) -> dict:
    return {
        "source": source,
        "volatility": volatility,
        "last_fetched_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    }

def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

# ─────────────────────────────────────────────
# DISCOVER ENTITIES
# ─────────────────────────────────────────────
def discover_entities(mode: str) -> dict:
    """
    Returns all entity lists the pipeline needs.
    Athletes carry structured metadata; rumors and injuries are
    separate signal lists that feed into the athlete enrichment step.
    """
    entities = {
        "narratives": [
            "Opening ceremony",
            "Closing ceremony",
            "Cultural program",
        ],

        # ── rumors: unconfirmed reports that matter for the chatbot.
        #   confidence: 0.0–1.0.  ≥0.8 means "almost confirmed".
        #   related_entity: ties the rumor to a narrative or athlete vector
        #     so the chatbot can link them when answering questions.
        #   status: "unconfirmed" | "confirmed" | "denied".
        #     confirmed rumors get promoted to narratives on the next run;
        #     denied rumors get their vectors deleted.
        "rumors": [
            {
                "id":              "bocelli_opening",
                "headline":        "Andrea Bocelli rumored to perform at Milano Cortina Opening Ceremony",
                "detail":          "Multiple Italian media outlets report that tenor Andrea Bocelli is in advanced discussions to headline the Opening Ceremony musical segment. No official confirmation from the Milano Cortina organizing committee yet.",
                "confidence":      0.75,
                "source":          "Italian sports press",
                "related_entity":  "Opening ceremony",
                "status":          "unconfirmed",
            },
        ],

        # ── injuries: fitness / injury status for athletes whose
        #   participation or performance could be affected.
        #   severity: low (minor, train through it) |
        #             moderate (may miss events or perform below peak) |
        #             high (likely withdrawal).
        #   event_impact: which of their scheduled events are at risk.
        "injuries": [
            {
                "athlete":        "Mikaela Shiffrin",
                "condition":      "Left ankle sprain — sustained during a World Cup giant slalom in Val d'Isère. Cleared for travel but training load reduced heading into the Games.",
                "severity":       "moderate",
                "status":         "training with modifications",
                "event_impact":   ["Women's downhill alpine skiing"],
                "source":         "USSA official statement",
            },
        ],

        # ── athletes ──
        "athletes": [
            {"name": "Mikaela Shiffrin",           "events": ["Women's downhill alpine skiing"],                          "favorite": True},
            {"name": "Yuzuru Hanyu",               "events": ["Men's figure skating free skate"],                         "favorite": True},
            {"name": "Ester Ledecká",              "events": ["Women's downhill alpine skiing", "Women's sprint"],        "favorite": True},
            {"name": "Jessie Diggins",             "events": ["Women's cross-country skiathlon"],                         "favorite": True},
            {"name": "John Shuster",               "events": ["Men's curling"],                                           "favorite": True},
            {"name": "Kendall Coyne Schofield",    "events": ["Women's 500m speed skating"],                              "favorite": True},
            {"name": "Danny O'Shea",               "events": ["Men's pairs figure skating"],                              "favorite": True},
            {"name": "Lee Stecklein",              "events": ["Women's ice hockey"],                                      "favorite": True},
            {"name": "Irene Schouten",             "events": ["Women's 500m speed skating"],                              "favorite": True},
            {"name": "Therese Johaug",             "events": ["Women's cross-country skiathlon"],                         "favorite": True},
        ],

        "events": [],
    }
    if mode == "LIVE_GAMES":
        entities["events"] = [
            "Women's downhill alpine skiing",
            "Men's figure skating free skate",
            "Women's ice hockey tournament",
            "Women's cross-country skiathlon",
            "Men's curling",
            "Women's 500m speed skating",
        ]
    return entities

# ─────────────────────────────────────────────
# FETCH STUBS
# ─────────────────────────────────────────────
def fetch_page(title: str) -> str:
    log.debug("fetch narrative: %s", title)
    return f"Latest updated content for {title}."

def fetch_athlete_bio(name: str) -> str:
    log.debug("fetch athlete bio: %s", name)
    bios = {
        "Lindsey Vonn":            "Three-time Olympic medalist in alpine skiing. Known for aggressive downhill technique and fierce rivalries.",
        "Nathan Chen":             "2022 Olympic gold medalist in men's figure skating. Holds the world record for most quadruple jumps in a single program.",
        "Marit Bjørgen":           "Most decorated female Winter Olympian in history. Dominant cross-country skier across four Olympics.",
        "Yuzuru Hanyu":            "Back-to-back Olympic gold medalist (2014, 2018). Pioneered the first competitive quad Axel attempt.",
        "Ester Ledecká":           "Czech athlete competing in both alpine skiing and sprint cycling — one of the most versatile Winter Olympians ever.",
        "Jessie Diggins":          "2022 Olympic gold medalist in cross-country skiing. First American woman to win Olympic cross-country gold.",
        "John Shuster":            "Led the USA to curling gold at 2018 PyeongChang. Veteran skip with three Olympic appearances.",
        "Kendall Coyne Schofield": "2018 Olympic gold medalist in speed skating. Known for blazing 500m times.",
        "Danny O'Shea":            "Rising star in pairs figure skating. Making his Olympic debut at Milano Cortina 2026.",
        "Lee Stecklein":           "USA women's ice hockey captain. Two-time Olympic gold medalist (2018, 2022).",
    }
    return bios.get(name, f"Athlete profile for {name}.")

def fetch_rumor(rumor: dict) -> dict:
    """
    In production this would re-scrape the source to check for updates.
    Stub returns the rumor as-is (simulates a fresh fetch with no status change).
    """
    log.debug("fetch rumor: %s", rumor["id"])
    return rumor

def fetch_injury(injury: dict) -> dict:
    """
    In production this would re-scrape team/league injury reports.
    Stub returns the injury as-is.
    """
    log.debug("fetch injury: %s", injury["athlete"])
    return injury

def fetch_event_results(event_name: str) -> list[dict]:
    log.debug("fetch event results: %s", event_name)
    STUBBED_RESULTS = {
        "Women's downhill alpine skiing": [
            {"rank": 1, "name": "Sara Hector",       "country": "SWE"},   # NOT a favorite → upset
            {"rank": 2, "name": "Mikaela Shiffrin",   "country": "USA"},   # favorite, silver
            {"rank": 3, "name": "Ester Ledecká",      "country": "CZE"},
        ],
        "Men's figure skating free skate": [
            {"rank": 1, "name": "Yuzuru Hanyu",      "country": "JPN"},   # favorite wins
            {"rank": 2, "name": "Kagiyama Kaito",    "country": "JPN"},
            {"rank": 3, "name": "Shoma Uno",         "country": "JPN"},   # NOT a favorite → upset
        ],
        "Women's ice hockey tournament": [
            {"rank": 1, "name": "Canada Women",      "country": "CAN"},   # NOT in favorites → upset
            {"rank": 2, "name": "USA Women",         "country": "USA"},
            {"rank": 3, "name": "Finland Women",     "country": "FIN"},
        ],
        "Women's cross-country skiathlon": [
            {"rank": 1, "name": "Jessie Diggins",    "country": "USA"},   # favorite wins
            {"rank": 2, "name": "Maja Dahlmeier",    "country": "GER"},
            {"rank": 3, "name": "Therese Johaug",    "country": "NOR"},   # favorite, bronze
        ],
        "Men's curling": [
            {"rank": 1, "name": "Sweden Men",        "country": "SWE"},   # NOT a favorite → upset
            {"rank": 2, "name": "USA Men",           "country": "USA"},
            {"rank": 3, "name": "Norway Men",        "country": "NOR"},
        ],
        "Women's 500m speed skating": [
            {"rank": 1, "name": "Irene Schouten",    "country": "NED"},   # favorite wins (defending champ)
            {"rank": 2, "name": "Kendall Coyne Schofield", "country": "USA"},
            {"rank": 3, "name": "Nao Kodaira",       "country": "JPN"},   # NOT a favorite → upset
        ],
    }
    return STUBBED_RESULTS.get(event_name, [
        {"rank": 1, "name": f"{event_name} Gold",   "country": "USA"},
        {"rank": 2, "name": f"{event_name} Silver", "country": "CAN"},
        {"rank": 3, "name": f"{event_name} Bronze", "country": "NOR"},
    ])

# ─────────────────────────────────────────────
# TRACKING — shared state built up across pipeline passes
# ─────────────────────────────────────────────
UPDATED_VECTORS      = []   # (vector_id, action)
EVENT_RESULTS_THIS_RUN = {} # event_name → [medalists]
INJURIES_THIS_RUN     = {}  # athlete_slug → injury dict
RUMORS_THIS_RUN       = []  # list of fetched rumor dicts

def upsert_document(vector_id: str, text: str, metadata: dict):
    action = upsert_vector(vector_id, text, metadata)
    UPDATED_VECTORS.append((vector_id, action))

# ─────────────────────────────────────────────
# UPSERT HELPERS — narratives
# ─────────────────────────────────────────────
def upsert_narrative(title: str, text: str):
    vid = f"page::{slug(title)}"
    upsert_document(vid, text, {
        "doc_type": "narrative",
        "title":    title,
        **freshness_metadata("wikipedia", "high"),
    })

# ─────────────────────────────────────────────
# UPSERT HELPERS — rumors
# ─────────────────────────────────────────────
def upsert_rumor(rumor: dict):
    """
    Writes a rumor vector.  The vector text is written so the LLM
    naturally hedges ("rumored", "unconfirmed") when it surfaces this.

    Lifecycle:
      unconfirmed → vector exists, confidence in metadata
      confirmed   → promoted: upsert into the related narrative instead,
                    then delete the rumor vector
      denied      → delete the rumor vector, no replacement
    """
    rid    = rumor["id"]
    status = rumor["status"]
    vid    = f"rumor::{rid}"

    if status == "confirmed":
        # Promote: merge into the related narrative
        log.warning("rumor CONFIRMED → promoting to narrative: %s", rid)
        related = rumor.get("related_entity", rid)
        promoted_text = (
            f"CONFIRMED: {rumor['headline']}\n"
            f"{rumor['detail']}\n"
            f"(Originally reported as unconfirmed; now confirmed by {rumor['source']}.)"
        )
        upsert_narrative(related, promoted_text)
        # Rumor vector no longer needed — mark for deletion in real Pinecone.
        # In this sim we just skip writing it.
        log.warning("rumor vector %s deleted (confirmed → narrative)", vid)
        return

    if status == "denied":
        log.warning("rumor DENIED → vector %s deleted", vid)
        return

    # status == "unconfirmed" — write the rumor vector
    confidence = rumor.get("confidence", 0.5)
    conf_label = "low" if confidence < 0.4 else "moderate" if confidence < 0.7 else "high"

    text = (
        f"RUMOR ({conf_label} confidence) — {rumor['headline']}\n"
        f"{rumor['detail']}\n"
        f"Source: {rumor['source']}.\n"
        f"Status: Unconfirmed as of this update. Treat as unverified."
    )
    upsert_document(vid, text, {
        "doc_type":       "rumor",
        "rumor_id":       rid,
        "confidence":     confidence,
        "conf_label":     conf_label,
        "status":         status,
        "related_entity": rumor.get("related_entity"),
        **freshness_metadata(rumor.get("source", "press"), "very_high"),
    })

# ─────────────────────────────────────────────
# UPSERT HELPERS — injuries
# ─────────────────────────────────────────────
def upsert_injury(injury: dict):
    """
    Writes an injury vector AND caches it in INJURIES_THIS_RUN so
    the athlete enrichment pass can stamp injury_risk onto the athlete vector.
    """
    athlete  = injury["athlete"]
    severity = injury["severity"]
    vid      = f"injury::{slug(athlete)}"

    # Cache for athlete enrichment
    INJURIES_THIS_RUN[slug(athlete)] = injury

    severity_icon = {"low": "🟡", "moderate": "🟠", "high": "🔴"}.get(severity, "⚪")

    text = (
        f"INJURY REPORT — {athlete}\n"
        f"Severity: {severity.upper()} {severity_icon}\n"
        f"Condition: {injury['condition']}\n"
        f"Status: {injury['status']}\n"
        f"Events at risk: {', '.join(injury.get('event_impact', []) or ['None identified'])}.\n"
        f"Source: {injury.get('source', 'unattributed')}."
    )
    upsert_document(vid, text, {
        "doc_type":     "injury",
        "athlete":      athlete,
        "severity":     severity,
        "status":       injury.get("status"),
        "event_impact": injury.get("event_impact", []),
        **freshness_metadata(injury.get("source", "team_report"), "very_high"),
    })

# ─────────────────────────────────────────────
# UPSERT HELPERS — events
# ─────────────────────────────────────────────
def upsert_event(event_name: str, medalists: list[dict]):
    vid = f"event::{slug(event_name)}"
    lines = [f"{m['rank']}. {m['name']} ({m['country']})" for m in medalists]
    text  = f"Event results — {event_name}\n" + "\n".join(lines)
    upsert_document(vid, text, {
        "doc_type":  "event_result",
        "event":     event_name,
        "medalists": medalists,
        **freshness_metadata("wikipedia", "low"),
    })
    EVENT_RESULTS_THIS_RUN[event_name] = medalists

# ─────────────────────────────────────────────
# UPSERT HELPERS — athletes (enriched)
# ─────────────────────────────────────────────
def upsert_athlete(athlete: dict):
    """
    Builds one rich vector per athlete by layering:
      1. Bio (fetched)
      2. Medal status (from EVENT_RESULTS_THIS_RUN)
      3. Injury status (from INJURIES_THIS_RUN — if present, adds a warning)
      4. Scheduled events + favorite flag
    """
    name      = athlete["name"]
    vid       = f"athlete::{slug(name)}"
    bio       = fetch_athlete_bio(name)
    favorite  = athlete.get("favorite", False)
    scheduled = athlete.get("events", [])

    # ── medal status ──
    medal_lines = []
    for event_name, medalists in EVENT_RESULTS_THIS_RUN.items():
        for m in medalists:
            if slug(m["name"]) == slug(name):
                ordinal = {1: "Gold", 2: "Silver", 3: "Bronze"}
                medal_lines.append(f"  {ordinal.get(m['rank'], '?')} — {event_name}")

    # ── injury status ──
    injury_info = INJURIES_THIS_RUN.get(slug(name))

    # ── assemble ──
    sections = [
        f"Athlete: {name}",
        f"Bio: {bio}",
        f"Favorite: {'Yes' if favorite else 'No'}",
        f"Scheduled events: {', '.join(scheduled) if scheduled else 'None'}",
    ]

    if medal_lines:
        sections.append("Medals this Games:\n" + "\n".join(medal_lines))
    else:
        sections.append("Medals this Games: None yet.")

    if injury_info:
        sev_icon = {"low": "🟡", "moderate": "🟠", "high": "🔴"}.get(injury_info["severity"], "⚪")
        sections.append(
            f"⚠️  INJURY FLAG {sev_icon} — {injury_info['severity'].upper()}\n"
            f"  Condition: {injury_info['condition']}\n"
            f"  Status: {injury_info['status']}\n"
            f"  Events at risk: {', '.join(injury_info.get('event_impact', []))}"
        )

    text = "\n".join(sections)

    upsert_document(vid, text, {
        "doc_type":         "athlete",
        "name":             name,
        "favorite":         favorite,
        "scheduled_events": scheduled,
        "has_medal":        len(medal_lines) > 0,
        "injury_risk":      injury_info["severity"] if injury_info else None,
        **freshness_metadata("wikipedia", "very_high"),
    })

# ─────────────────────────────────────────────
# UPSERT HELPERS — upsets
# ─────────────────────────────────────────────
def upsert_upset(event_name: str, medalist: dict):
    name    = medalist["name"]
    country = medalist["country"]
    rank    = medalist["rank"]
    ordinal = {1: "Gold", 2: "Silver", 3: "Bronze"}.get(rank, "Medal")

    vid  = f"upset::{slug(event_name)}_{slug(name)}"
    text = (
        f"UPSET — {event_name}\n"
        f"{name} ({country}) won {ordinal} — an unexpected result.\n"
        f"{name} was not among the pre-Games favorites for this event.\n"
        f"This is one of the surprise storylines of Milano Cortina 2026."
    )
    upsert_document(vid, text, {
        "doc_type": "upset",
        "event":    event_name,
        "athlete":  name,
        "country":  country,
        "medal":    ordinal,
        **freshness_metadata("results_scraper", "very_high"),
    })
    log.info("UPSET: %s (%s) — %s in %s", name, country, ordinal, event_name)

# ─────────────────────────────────────────────
# UPSET DETECTION
# ─────────────────────────────────────────────
def detect_upsets():
    """
    Skips TEAM_EVENTS — collective names like "USA Women" can't be
    checked against an individual-athlete favorites roster.
    For individual events: any medalist not in KNOWN_FAVORITES gets
    a dedicated upset vector.
    """
    log.info("── upset detection (individual) ──")
    upsets_found = 0
    for event_name, medalists in EVENT_RESULTS_THIS_RUN.items():
        if event_name in TEAM_EVENTS:
            log.debug("skipping team event: %s", event_name)
            continue
        for m in medalists:
            if slug(m["name"]) not in KNOWN_FAVORITES:
                upsert_upset(event_name, m)
                upsets_found += 1
    if upsets_found == 0:
        log.debug("no individual upsets this run")
    return upsets_found

# ─────────────────────────────────────────────
# COUNTRY-LEVEL UPSET DETECTION
# ─────────────────────────────────────────────
def upsert_country_upset(country: str, signal_type: str, detail: str, metadata_extra: dict):
    """
    Writes a country_upset:: vector.  signal_type is one of:
      team_event  — a favored country lost gold in a team event
      surge       — a country exceeded its historical gold baseline
      shutout     — a favored country failed to medal in an expected event
    """
    vid  = f"country_upset::{signal_type}_{slug(country)}"
    # If there's already a vector for this country+signal (e.g. multiple
    # shutouts for the same country), append a unique event tag.
    if "event" in metadata_extra:
        vid = f"country_upset::{signal_type}_{slug(country)}_{slug(metadata_extra['event'])}"

    text = (
        f"COUNTRY UPSET ({signal_type.replace('_', ' ').upper()}) — {country}\n"
        f"{detail}\n"
        f"This is a notable storyline at Milano Cortina 2026."
    )
    upsert_document(vid, text, {
        "doc_type":    "country_upset",
        "country":     country,
        "signal_type": signal_type,
        **metadata_extra,
        **freshness_metadata("results_scraper", "very_high"),
    })
    log.info("COUNTRY UPSET (%s): %s", signal_type, country)


def detect_country_upsets():
    """
    Three independent signals, all built from EVENT_RESULTS_THIS_RUN:

    1. TEAM EVENT — for each event in TEAM_EVENT_FAVORITES, check whether
       the favored country actually won gold.  If not, the country that
       DID win gold gets a country_upset vector.

    2. SURGE — tally all golds across every event this run.  Any country
       whose gold count exceeds HISTORICAL_GOLD_BASELINE by more than
       COUNTRY_SURGE_THRESHOLD gets a surge vector.

    3. SHUTOUT — for each event in EVENT_EXPECTED_COUNTRIES, check whether
       every expected country appears at least once on the podium.  If an
       expected country is entirely absent, it gets a shutout vector.
    """
    log.info("── upset detection (country) ──")
    country_upsets_found = 0

    # ── build medal tally (golds only for surge; full for shutout) ──
    gold_tally = {}   # country → int
    for event_name, medalists in EVENT_RESULTS_THIS_RUN.items():
        for m in medalists:
            c = m["country"]
            if m["rank"] == 1:
                gold_tally[c] = gold_tally.get(c, 0) + 1

    # ── Signal 1: team event upsets ──
    log.debug("[1/3] team event check")
    for event_name, favored_country in TEAM_EVENT_FAVORITES.items():
        if event_name not in EVENT_RESULTS_THIS_RUN:
            continue
        gold_winner = next(
            (m for m in EVENT_RESULTS_THIS_RUN[event_name] if m["rank"] == 1),
            None
        )
        if gold_winner and gold_winner["country"] != favored_country:
            actual_country = gold_winner["country"]
            upsert_country_upset(
                country=actual_country,
                signal_type="team_event",
                detail=(
                    f"{actual_country} won gold in {event_name}, "
                    f"defeating {favored_country} who were the pre-Games favorites. "
                    f"Winner: {gold_winner['name']}."
                ),
                metadata_extra={
                    "event":            event_name,
                    "favored_country":  favored_country,
                    "winner_name":      gold_winner["name"],
                },
            )
            country_upsets_found += 1
        else:
            log.debug("%s: %s won as expected", event_name, favored_country)

    # ── Signal 2: country surge ──
    log.debug("[2/3] surge check")
    for country, golds in sorted(gold_tally.items(), key=lambda x: -x[1]):
        baseline = HISTORICAL_GOLD_BASELINE.get(country, 0)
        delta    = golds - baseline
        if delta > COUNTRY_SURGE_THRESHOLD:
            upsert_country_upset(
                country=country,
                signal_type="surge",
                detail=(
                    f"{country} has won {golds} gold medal{'s' if golds != 1 else ''} "
                    f"— {delta} more than the {baseline} expected based on historical performance. "
                    f"A genuine surprise run at these Games."
                ),
                metadata_extra={
                    "golds_actual":   golds,
                    "golds_baseline": baseline,
                    "delta":          delta,
                },
            )
            country_upsets_found += 1
        else:
            log.debug("%s: %d golds (baseline %d, Δ%+d) — within threshold", country, golds, baseline, delta)

    # ── Signal 3: shutouts ──
    log.debug("[3/3] shutout check")
    for event_name, expected_countries in EVENT_EXPECTED_COUNTRIES.items():
        if event_name not in EVENT_RESULTS_THIS_RUN:
            continue
        actual_countries = {m["country"] for m in EVENT_RESULTS_THIS_RUN[event_name]}
        for ec in expected_countries:
            if ec not in actual_countries:
                upsert_country_upset(
                    country=ec,
                    signal_type="shutout",
                    detail=(
                        f"{ec} was expected to medal in {event_name} "
                        f"but did not appear on the podium — a significant absence."
                    ),
                    metadata_extra={"event": event_name},
                )
                country_upsets_found += 1
            else:
                log.debug("%s medaled in %s", ec, event_name)

    if country_upsets_found == 0:
        log.debug("no country-level upsets this run")
    return country_upsets_found

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
def summarize_updates(updated_list: list[tuple[str, str]]) -> dict:
    summary = {
        "narratives":     [],
        "rumors":         [],
        "injuries":       [],
        "athletes":       [],
        "events":         [],
        "upsets":         [],
        "country_upsets": [],
    }
    for vid, action in updated_list:
        record = f"{vid} ({action})"
        # country_upset:: checked before upset:: — longer prefix, same start
        if   vid.startswith("athlete::"):        summary["athletes"].append(record)
        elif vid.startswith("event::"):          summary["events"].append(record)
        elif vid.startswith("country_upset::"):  summary["country_upsets"].append(record)
        elif vid.startswith("upset::"):          summary["upsets"].append(record)
        elif vid.startswith("rumor::"):          summary["rumors"].append(record)
        elif vid.startswith("injury::"):         summary["injuries"].append(record)
        else:                                    summary["narratives"].append(record)
    return summary

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    mode = resolve_mode()
    log.info("=" * 60)
    log.info("PIPELINE MODE: %s", mode)
    log.info("run started: %s", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    log.info("=" * 60)

    if mode == "DORMANT":
        log.info("DORMANT — exiting without updates")
        return

    # Connect to Pinecone when the key is available (CI / production).
    # Absent key → in-memory VECTOR_STORE only (tests / local dev).
    if _os.getenv("PINECONE_API_KEY"):
        _init_pinecone()
    else:
        log.info("PINECONE_API_KEY not set — using in-memory store only")

    entities = discover_entities(mode)

    # ── 1. Narratives ──
    log.info("── narratives ──")
    for page in entities["narratives"]:
        text = fetch_page(page)
        upsert_narrative(page, text)
        time.sleep(0.1)

    # ── 2. Rumors ──
    #    Runs before athletes so that if a rumor is about an athlete
    #    (future expansion), the athlete pass could reference it.
    log.info("── rumors ──")
    for rumor in entities["rumors"]:
        fresh = fetch_rumor(rumor)
        RUMORS_THIS_RUN.append(fresh)
        upsert_rumor(fresh)
        time.sleep(0.1)

    # ── 3. Injuries ──
    #    Must run before athletes — populates INJURIES_THIS_RUN
    #    which the athlete enrichment step reads.
    log.info("── injuries ──")
    for injury in entities["injuries"]:
        fresh = fetch_injury(injury)
        upsert_injury(fresh)
        time.sleep(0.1)

    # ── 4. Events (LIVE only) ──
    #    Must run before athletes — populates EVENT_RESULTS_THIS_RUN.
    if mode == "LIVE_GAMES":
        log.info("── events ──")
        for event_name in entities["events"]:
            medalists = fetch_event_results(event_name)
            upsert_event(event_name, medalists)
            time.sleep(0.1)

    # ── 5. Athletes (enriched: bio + medals + injuries) ──
    log.info("── athletes ──")
    for athlete in entities["athletes"]:
        upsert_athlete(athlete)
        time.sleep(0.1)

    # ── 6. Upset detection — individual (LIVE only) ──
    if mode == "LIVE_GAMES":
        log.info("── upset detection (individual) ──")
        detect_upsets()

    # ── 7. Upset detection — country (LIVE only) ──
    #    Three signals: team_event, surge, shutout.
    #    All built from EVENT_RESULTS_THIS_RUN after events are cached.
    if mode == "LIVE_GAMES":
        log.info("── upset detection (country) ──")
        detect_country_upsets()

    # ── Summary ──
    summary = summarize_updates(UPDATED_VECTORS)
    log.info("=" * 60)
    log.info("UPDATE SUMMARY")
    log.info("=" * 60)
    for key, items in summary.items():
        flag = "🚨" if key in ("upsets", "country_upsets") and items else "  "
        log.info("%s %s (%d):", flag, key.upper(), len(items))
        for item in items:
            log.info("   - %s", item)

    log.info("total vectors touched: %d", len(UPDATED_VECTORS))
    log.info("pipeline run complete")

# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
