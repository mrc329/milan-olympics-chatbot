"""
milan2026_pipeline.py (NAMESPACE-AWARE VERSION)
================================================
Winter Olympics-focused pipeline for Milan 2026.

UPDATED: Routes vectors to appropriate Pinecone namespaces:
  - athletes/     → athlete profiles (enriched with medals, injuries)
  - events/       → event results, upsets, country_upsets
  - narratives/   → pages, rumors, injuries

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
                           dedicated upset vector.
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
import os as _os
logging.basicConfig(
    level=getattr(logging, _os.getenv("PIPELINE_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    force=True,
)
log = logging.getLogger("milan2026_pipeline")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GAMES_START = datetime(2026, 2, 5, tzinfo=timezone.utc)
GAMES_END   = datetime(2026, 2, 22, 23, 59, tzinfo=timezone.utc)

FRESHNESS_SLA_MINUTES = {
    "narrative":      60,
    "rumor":          20,
    "injury":         15,
    "athlete":        30,
    "event":          15,
    "upset":           5,
    "country_upset":   5,
}

KNOWN_FAVORITES = {
    "yuzuru_hanyu",
    "ester_ledeck",
    "jessie_diggins",
    "john_shuster",
    "kendall_coyne_schofield",
    "danny_o_shea",
    "lee_stecklein",
    "mikaela_shiffrin",
    "irene_schouten",
    "therese_johaug",
}

TEAM_EVENTS = {
    "Women's ice hockey tournament",
    "Men's curling",
}

TEAM_EVENT_FAVORITES = {
    "Women's ice hockey tournament": "USA",
    "Men's curling":                 "USA",
}

HISTORICAL_GOLD_BASELINE = {
    "USA": 1,
    "NOR": 1,
    "JPN": 1,
    "SWE": 0,
    "NED": 0,
    "CAN": 0,
    "CZE": 0,
    "FIN": 0,
}

COUNTRY_SURGE_THRESHOLD = 1

EVENT_EXPECTED_COUNTRIES = {
    "Women's downhill alpine skiing":      {"USA", "CZE"},
    "Men's figure skating free skate":     {"JPN"},
    "Women's ice hockey tournament":       {"USA", "CAN"},
    "Women's cross-country skiathlon":     {"NOR", "USA"},
    "Men's curling":                       {"USA"},
    "Women's 500m speed skating":          {"USA", "NED"},
}

# ─────────────────────────────────────────────
# PINECONE + EMBEDDING MODEL
# ─────────────────────────────────────────────
INDEX_NAME   = "milan-2026-olympics"
MODEL_NAME   = "all-MiniLM-L6-v2"

_pinecone_index = None
_embedder       = None

def _init_pinecone():
    """Connect to Pinecone and load the embedding model."""
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
VECTOR_STORE = {}

def upsert_vector(vector_id: str, text: str, metadata: dict) -> str:
    """
    Upsert a vector to the appropriate namespace based on vector_id prefix.
    
    Namespace routing:
      athlete::*          → athletes
      event::*            → events
      upset::*            → events
      country_upset::*    → events
      page::*             → narratives
      rumor::*            → narratives
      injury::*           → narratives
    """
    action = "inserted" if vector_id not in VECTOR_STORE else "updated"

    if _pinecone_index is not None:
        # Determine namespace from vector_id prefix
        if vector_id.startswith("athlete::"):
            namespace = "athletes"
        elif vector_id.startswith("event::") or vector_id.startswith("upset::") or vector_id.startswith("country_upset::"):
            namespace = "events"
        elif vector_id.startswith("page::") or vector_id.startswith("rumor::") or vector_id.startswith("injury::"):
            namespace = "narratives"
        else:
            namespace = "narratives"  # default fallback
        
        # Embed and upsert
        embedding = _embedder.encode(text).tolist()
        metadata_with_text = {**metadata, "text": text, "namespace": namespace}
        _pinecone_index.upsert(
            vectors=[{
                "id": vector_id,
                "values": embedding,
                "metadata": metadata_with_text,
            }],
            namespace=namespace
        )
        
        log.debug(f"upserted to namespace '{namespace}': {vector_id}")

    # Always mirror to in-memory store (tests read it for assertions)
    VECTOR_STORE[vector_id] = {"text": text, "metadata": metadata}
    log.info("upsert %-8s → %s (namespace: %s)", action.upper(), vector_id, 
             metadata.get('namespace', 'in-memory-only'))
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
    entities = {
        "narratives": [
            "Opening ceremony",
            "Closing ceremony",
            "Cultural program",
        ],

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
        "Mikaela Shiffrin":          "Three-time Olympic medalist in alpine skiing. Known for aggressive downhill technique and fierce rivalries.",
        "Yuzuru Hanyu":              "Back-to-back Olympic gold medalist (2014, 2018). Pioneered the first competitive quad Axel attempt.",
        "Ester Ledecká":             "Czech athlete competing in both alpine skiing and sprint cycling — one of the most versatile Winter Olympians ever.",
        "Jessie Diggins":            "2022 Olympic gold medalist in cross-country skiing. First American woman to win Olympic cross-country gold.",
        "John Shuster":              "Led the USA to curling gold at 2018 PyeongChang. Veteran skip with three Olympic appearances.",
        "Kendall Coyne Schofield":   "2018 Olympic gold medalist in speed skating. Known for blazing 500m times.",
        "Danny O'Shey":              "Rising star in pairs figure skating. Making his Olympic debut at Milano Cortina 2026.",
        "Lee Stecklein":             "USA women's ice hockey captain. Two-time Olympic gold medalist (2018, 2022).",
        "Irene Schouten":            "Defending Olympic champion in multiple speed skating events. Dominant distance skater.",
        "Therese Johaug":            "Cross-country legend returning after missing 2022. Chasing more gold.",
    }
    return bios.get(name, f"Athlete profile for {name}.")

def fetch_rumor(rumor: dict) -> dict:
    log.debug("fetch rumor: %s", rumor["id"])
    return rumor

def fetch_injury(injury: dict) -> dict:
    log.debug("fetch injury: %s", injury["athlete"])
    return injury

def fetch_event_results(event_name: str) -> list[dict]:
    log.debug("fetch event results: %s", event_name)
    STUBBED_RESULTS = {
        "Women's downhill alpine skiing": [
            {"rank": 1, "name": "Sara Hector",       "country": "SWE"},
            {"rank": 2, "name": "Mikaela Shiffrin",   "country": "USA"},
            {"rank": 3, "name": "Ester Ledecká",      "country": "CZE"},
        ],
        "Men's figure skating free skate": [
            {"rank": 1, "name": "Yuzuru Hanyu",      "country": "JPN"},
            {"rank": 2, "name": "Kagiyama Kaito",    "country": "JPN"},
            {"rank": 3, "name": "Shoma Uno",         "country": "JPN"},
        ],
        "Women's ice hockey tournament": [
            {"rank": 1, "name": "Canada Women",      "country": "CAN"},
            {"rank": 2, "name": "USA Women",         "country": "USA"},
            {"rank": 3, "name": "Finland Women",     "country": "FIN"},
        ],
        "Women's cross-country skiathlon": [
            {"rank": 1, "name": "Jessie Diggins",    "country": "USA"},
            {"rank": 2, "name": "Maja Dahlmeier",    "country": "GER"},
            {"rank": 3, "name": "Therese Johaug",    "country": "NOR"},
        ],
        "Men's curling": [
            {"rank": 1, "name": "Sweden Men",        "country": "SWE"},
            {"rank": 2, "name": "USA Men",           "country": "USA"},
            {"rank": 3, "name": "Norway Men",        "country": "NOR"},
        ],
        "Women's 500m speed skating": [
            {"rank": 1, "name": "Irene Schouten",    "country": "NED"},
            {"rank": 2, "name": "Kendall Coyne Schofield", "country": "USA"},
            {"rank": 3, "name": "Nao Kodaira",       "country": "JPN"},
        ],
    }
    return STUBBED_RESULTS.get(event_name, [
        {"rank": 1, "name": f"{event_name} Gold",   "country": "USA"},
        {"rank": 2, "name": f"{event_name} Silver", "country": "CAN"},
        {"rank": 3, "name": f"{event_name} Bronze", "country": "NOR"},
    ])

# ─────────────────────────────────────────────
# TRACKING
# ─────────────────────────────────────────────
UPDATED_VECTORS      = []
EVENT_RESULTS_THIS_RUN = {}
INJURIES_THIS_RUN     = {}
RUMORS_THIS_RUN       = []

def upsert_document(vector_id: str, text: str, metadata: dict):
    action = upsert_vector(vector_id, text, metadata)
    UPDATED_VECTORS.append((vector_id, action))

# ─────────────────────────────────────────────
# UPSERT HELPERS
# ─────────────────────────────────────────────
def upsert_narrative(title: str, text: str):
    vid = f"page::{slug(title)}"
    upsert_document(vid, text, {
        "doc_type": "narrative",
        "title":    title,
        **freshness_metadata("wikipedia", "high"),
    })

def upsert_rumor(rumor: dict):
    rid    = rumor["id"]
    status = rumor["status"]
    vid    = f"rumor::{rid}"

    if status == "confirmed":
        log.warning("rumor CONFIRMED → promoting to narrative: %s", rid)
        related = rumor.get("related_entity", rid)
        promoted_text = (
            f"CONFIRMED: {rumor['headline']}\n"
            f"{rumor['detail']}\n"
            f"(Originally reported as unconfirmed; now confirmed by {rumor['source']}.)"
        )
        upsert_narrative(related, promoted_text)
        log.warning("rumor vector %s deleted (confirmed → narrative)", vid)
        return

    if status == "denied":
        log.warning("rumor DENIED → vector %s deleted", vid)
        return

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

def upsert_injury(injury: dict):
    athlete  = injury["athlete"]
    severity = injury["severity"]
    vid      = f"injury::{slug(athlete)}"

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

def upsert_athlete(athlete: dict):
    name      = athlete["name"]
    vid       = f"athlete::{slug(name)}"
    bio       = fetch_athlete_bio(name)
    favorite  = athlete.get("favorite", False)
    scheduled = athlete.get("events", [])

    medal_lines = []
    for event_name, medalists in EVENT_RESULTS_THIS_RUN.items():
        for m in medalists:
            if slug(m["name"]) == slug(name):
                ordinal = {1: "Gold", 2: "Silver", 3: "Bronze"}
                medal_lines.append(f"  {ordinal.get(m['rank'], '?')} — {event_name}")

    injury_info = INJURIES_THIS_RUN.get(slug(name))

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

    meta = {
        "doc_type":         "athlete",
        "name":             name,
        "favorite":         favorite,
        "scheduled_events": scheduled,
        "has_medal":        len(medal_lines) > 0,
        **freshness_metadata("wikipedia", "very_high"),
    }
    if injury_info:
        meta["injury_risk"] = injury_info["severity"]

    upsert_document(vid, text, meta)

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

def upsert_country_upset(country: str, signal_type: str, detail: str, metadata_extra: dict):
    vid  = f"country_upset::{signal_type}_{slug(country)}"
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

# ─────────────────────────────────────────────
# UPSET DETECTION
# ─────────────────────────────────────────────
def detect_upsets():
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

def detect_country_upsets():
    log.info("── upset detection (country) ──")
    country_upsets_found = 0

    gold_tally = {}
    for event_name, medalists in EVENT_RESULTS_THIS_RUN.items():
        for m in medalists:
            c = m["country"]
            if m["rank"] == 1:
                gold_tally[c] = gold_tally.get(c, 0) + 1

    # Signal 1: team event
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

    # Signal 2: surge
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

    # Signal 3: shutout
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

    if _os.getenv("PINECONE_API_KEY"):
        _init_pinecone()
    else:
        log.info("PINECONE_API_KEY not set — using in-memory store only")

    entities = discover_entities(mode)

    # 1. Narratives
    log.info("── narratives ──")
    for page in entities["narratives"]:
        text = fetch_page(page)
        upsert_narrative(page, text)
        time.sleep(0.1)

    # 2. Rumors
    log.info("── rumors ──")
    for rumor in entities["rumors"]:
        fresh = fetch_rumor(rumor)
        RUMORS_THIS_RUN.append(fresh)
        upsert_rumor(fresh)
        time.sleep(0.1)

    # 3. Injuries
    log.info("── injuries ──")
    for injury in entities["injuries"]:
        fresh = fetch_injury(injury)
        upsert_injury(fresh)
        time.sleep(0.1)

    # 4. Events (LIVE only)
    if mode == "LIVE_GAMES":
        log.info("── events ──")
        for event_name in entities["events"]:
            medalists = fetch_event_results(event_name)
            upsert_event(event_name, medalists)
            time.sleep(0.1)

    # 5. Athletes
    log.info("── athletes ──")
    for athlete in entities["athletes"]:
        upsert_athlete(athlete)
        time.sleep(0.1)

    # 6. Upset detection (LIVE only)
    if mode == "LIVE_GAMES":
        detect_upsets()

    # 7. Country upset detection (LIVE only)
    if mode == "LIVE_GAMES":
        detect_country_upsets()

    # Summary
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

if __name__ == "__main__":
    main()
