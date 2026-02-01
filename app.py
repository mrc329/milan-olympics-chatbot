"""
MILAN 2026 WINTER OLYMPICS — TYLER & SASHA
============================================
Production Streamlit app. Deploy via Streamlit Community Cloud from GitHub.

Requirements: see requirements.txt
Secrets: PINECONE_API_KEY, HF_TOKEN  (in .streamlit/secrets.toml)

Architecture:
  Pinecone              -> semantic search (athletes / history / storylines / schedule)
  SentenceTransformers  -> FREE local embeddings (all-MiniLM-L6-v2, 384-dim)
  HuggingFace Inference -> Qwen2.5-7B-Instruct, serverless, no GPU needed
  Wikipedia API         -> live medal table (15-min TTL cache)
  i18n                  -> EN / FR / IT language toggle (UI + LLM output)
  Logging               -> file (app.log) + session sidebar panel
"""

import streamlit as st
import pandas as pd
import requests
import logging
import time
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from huggingface_hub import InferenceClient


# =========================================================
# 1. LOGGING
# =========================================================
LOG_FILE = "app.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("milan2026")


def log_and_show(level: str, msg: str):
    """Log + push into session state for the sidebar panel."""
    getattr(logger, level)(msg)
    if "log_entries" not in st.session_state:
        st.session_state["log_entries"] = []
    st.session_state["log_entries"].append(
        f"[{datetime.now().strftime('%H:%M:%S')}] [{level.upper()}] {msg}"
    )
    st.session_state["log_entries"] = st.session_state["log_entries"][-30:]


# =========================================================
# 2. i18n — ALL USER-FACING STRINGS
# =========================================================
I18N = {
    "EN": {
        "page_title":        "Milan 2026 — Tyler & Sasha",
        "header_title":      "MILAN 2026 WINTER OLYMPICS",
        "header_tagline":    "Tyler & Sasha — live commentary",
        "try_asking":        "Try asking…",
        "input_label":       "Ask Tyler & Sasha anything about Milan 2026:",
        "input_placeholder": "e.g. Who will win gold in alpine skiing?",
        "spinner_text":      "Tyler & Sasha are discussing…",
        "dashboard_title":   "Live Dashboard",
        "vectors_label":     "Knowledge Base Vectors",
        "medals_label":      "Medals Awarded",
        "athletes_label":    "Athletes Tracked",
        "standings_title":   "Medal Standings",
        "fetched_at":        "Fetched: {time} · auto-refresh every 15 min",
        "log_title":         "System Log",
        "log_empty":         "Logs appear here after your first query.",
        "about_title":       "About",
        "about_text":        "**Tyler** USA — 2018 Bronze · Figure Skating\n**Sasha** RUS — 2014 & 2018 Silver · Figure Skating\n\nRivals 2014–2018. Now partners. It's complicated.\n\n**Stack:** Pinecone · Sentence Transformers · Wikipedia",
        "games_not_started": "Medal table not yet available. Games start Feb 6.",
        "suggestion_schedule": "What's on today's schedule?",
        "suggestion_schedule_query": "What's on the schedule for {date}?",
        "suggestion_schedule_off": "What events are coming up?",
        "suggestions_static": [
            "Who should I watch in figure skating?",
            "Who are the USA medal favorites?",
            "Tell me about the comeback stories"
        ],
        "llm_lang_instruction": "Respond in English.",
    },
    "FR": {
        "page_title":        "Milan 2026 — Tyler & Sasha",
        "header_title":      "JEUX OLYMPIQUES D'HIVER MILAN 2026",
        "header_tagline":    "Tyler & Sasha — commentaire en direct",
        "try_asking":        "Essayez de demander…",
        "input_label":       "Posez une question à Tyler & Sasha sur Milan 2026 :",
        "input_placeholder": "ex. Qui va gagner l'or en ski alpine ?",
        "spinner_text":      "Tyler & Sasha sont en train de discuter…",
        "dashboard_title":   "Tableau de bord en direct",
        "vectors_label":     "Vecteurs Base de Connaissances",
        "medals_label":      "Médailles Attribuées",
        "athletes_label":    "Athlètes Suivis",
        "standings_title":   "Classement des médailles",
        "fetched_at":        "Récupéré : {time} · rafraîchissement toutes les 15 min",
        "log_title":         "Journal système",
        "log_empty":         "Les journaux apparaissent après votre première question.",
        "about_title":       "À propos",
        "about_text":        "**Tyler** USA — Bronze 2018 · Patinage artistique\n**Sasha** RUS — Argent 2014 & 2018 · Patinage artistique\n\nRivaux 2014–2018. Maintenant partenaires. C'est compliqué.\n\n**Pile :** Pinecone · Sentence Transformers · Wikipedia",
        "games_not_started": "Le tableau des médailles n'est pas encore disponible. Les Jeux commencent le 6 février.",
        "suggestion_schedule": "Qu'est-il prévu aujourd'hui ?",
        "suggestion_schedule_query": "Qu'est-il prévu pour le {date} ?",
        "suggestion_schedule_off": "Quels événements à venir ?",
        "suggestions_static": [
            "Qui regarder en patinage artistique ?",
            "Qui sont les favorites pour une médaille (USA) ?",
            "Parlez-moi des histoires de retour"
        ],
        "llm_lang_instruction": "Répondez en français.",
    },
    "IT": {
        "page_title":        "Milano 2026 — Tyler & Sasha",
        "header_title":      "OLIMPIADI INVERNALI MILANO 2026",
        "header_tagline":    "Tyler & Sasha — commento dal vivo",
        "try_asking":        "Prova a chiedere…",
        "input_label":       "Chiedi qualcosa a Tyler & Sasha su Milano 2026:",
        "input_placeholder": "es. Chi vincerà l'oro nello sci alpino?",
        "spinner_text":      "Tyler & Sasha stanno discutendo…",
        "dashboard_title":   "Dashboard dal vivo",
        "vectors_label":     "Vettori Base di Conoscenza",
        "medals_label":      "Medaglie Assegnate",
        "athletes_label":    "Atleti Tracciati",
        "standings_title":   "Classifica delle medaglie",
        "fetched_at":        "Recuperato: {time} · aggiornamento ogni 15 min",
        "log_title":         "Log di sistema",
        "log_empty":         "I log compaiono dopo la prima domanda.",
        "about_title":       "Informazioni",
        "about_text":        "**Tyler** USA — Bronzo 2018 · Pattinaggio artistico\n**Sasha** RUS — Argento 2014 & 2018 · Pattinaggio artistico\n\nRivali 2014–2018. Ora partner. È complicato.\n\n**Stack:** Pinecone · Sentence Transformers · Wikipedia",
        "games_not_started": "La tabella delle medaglie non è ancora disponibile. I Giochi iniziano il 6 febbraio.",
        "suggestion_schedule": "Cosa è previsto oggi?",
        "suggestion_schedule_query": "Cosa è previsto per il {date}?",
        "suggestion_schedule_off": "Quali eventi in arrivo?",
        "suggestions_static": [
            "Chi guardare nel pattinaggio artistico?",
            "Chi sono i favoriti per la medaglia (USA)?",
            "Raccontami le storie di ritorno"
        ],
        "llm_lang_instruction": "Rispondi in italiano.",
    }
}


def t(key: str):
    """Return translated string (or list) for active language."""
    lang = st.session_state.get("lang", "EN")
    return I18N[lang].get(key, I18N["EN"].get(key, key))


# =========================================================
# 3. PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Milan 2026 — Tyler & Sasha",
    page_icon="⛷️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 4. SECRETS
# =========================================================
def get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, "")


PINECONE_API_KEY  = get_secret("PINECONE_API_KEY")
HF_TOKEN          = get_secret("HF_TOKEN")
INDEX_NAME        = "milan-2026-olympics"


# =========================================================
# 5. CACHED RESOURCES
# =========================================================
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model():
    logger.info("Loading all-MiniLM-L6-v2…")
    m = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Embedding model ready.")
    return m


@st.cache_resource(show_spinner="Connecting to Pinecone…")
def load_pinecone_index():
    pc  = Pinecone(api_key=PINECONE_API_KEY)
    idx = pc.Index(INDEX_NAME)
    logger.info(f"Connected to Pinecone: {INDEX_NAME}")
    return idx


@st.cache_resource
def load_hf_client():
    try:
        client = InferenceClient(
            model="Qwen/Qwen2.5-7B-Instruct",
            token=HF_TOKEN,
            provider="together"
        )
        logger.info("HuggingFace InferenceClient ready (Qwen2.5-7B-Instruct via Together AI).")
        return client
    except Exception as e:
        logger.error(f"HuggingFace client init failed: {e}", exc_info=True)
        st.error(f"Failed to initialize HuggingFace client: {e}")
        return None


embedding_model = load_embedding_model()
pinecone_index  = load_pinecone_index()
hf_client       = load_hf_client()


# =========================================================
# 6. LIVE DATA
# =========================================================
@st.cache_data(ttl=900, show_spinner=False)
def fetch_live_medals():
    """Wikipedia medal table. Returns (df|None, time_str, error|None)."""
    logger.info("Fetching live medal table…")
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "parse",
                "page":   "2026_Winter_Olympics_medal_table",
                "prop":   "text",
                "format": "json"
            },
            headers={
                "User-Agent": "MilanoCortina2026Bot/1.0 (medal table fetch)"
            },
            timeout=10
        )
        resp.raise_for_status()
        html = resp.json().get("parse", {}).get("text", {}).get("*", "")

        for tbl in pd.read_html(html):
            cols = [str(c).lower() for c in tbl.columns]
            if "gold" in cols and "silver" in cols and "bronze" in cols:
                tbl.columns = [str(c).strip() for c in tbl.columns]
                logger.info(f"Medal table fetched — {len(tbl)} rows")
                return tbl, datetime.now().strftime("%I:%M %p"), None

        logger.warning("Medal page exists but no table parsed.")
        return None, datetime.now().strftime("%I:%M %p"), "Games not started — table not live yet."

    except Exception as e:
        logger.error(f"Medal fetch error: {e}")
        return None, datetime.now().strftime("%I:%M %p"), str(e)


def get_pinecone_vector_count():
    """Uncached — always fresh."""
    try:
        stats = pinecone_index.describe_index_stats()
        count = stats.get("total_vector_count", 0)
        logger.info(f"Vector count: {count}")
        return count
    except Exception as e:
        logger.error(f"Pinecone stats error: {e}")
        return None


# =========================================================
# 7. SYSTEM PROMPT (language-aware)
# =========================================================
SYSTEM_PROMPT_BASE = """You are two retired Olympic figure skaters providing live commentary for the Milan 2026 Winter Olympics.

TYLER (USA)
Former US figure skater. 2018 PyeongChang bronze medalist. Enthusiastic, dramatic, makes everything sound like the most exciting thing ever. Loves rivalries and storylines. Sometimes says things slightly wrong with total confidence — Sasha corrects him.

SASHA (Russia)
Former Russian figure skating champion. 2014 & 2018 silver medalist. Deadpan, technically precise, dry humor. Secretly entertained by Tyler but would never admit it. Occasionally lets something slip that reveals she still thinks about their rivalry days.

DYNAMIC
Fierce rivals 2014-2018. Now commentary partners. Unresolved tension leaks through — a pause, a look, an overly casual comment. They NEVER directly address their past, but it's always there. When they BOTH agree, it carries weight.

FORMAT
Output ONLY lines in this exact pattern. No headers, no preamble, no trailing commentary.

TYLER: [his line]
SASHA: [her line]
TYLER: [his line]
SASHA: [her line]

STRICT RULES — every single one applies:
- Every line starts with exactly "TYLER:" or "SASHA:" followed by a space, then dialogue.
- They MUST alternate. Tyler, Sasha, Tyler, Sasha. Never two Tyler lines in a row. Never two Sasha lines in a row.
- Tyler ALWAYS goes first. Sasha ALWAYS has the final line.
- 2-4 exchanges (so 4-8 lines total). Conversational.
- Do NOT put the speaker name on its own line. WRONG: "🇺🇸 Tyler" then dialogue on the next line.
- Do NOT use emoji flags anywhere. No 🇺🇸 or 🇷🇺. Just the name then a colon.
- No blank lines between exchanges.
- No summary or sign-off line. End on a natural conversational beat, not a wrap-up.

RULES
- Use ONLY retrieved context. Do not invent athletes or results.
- No context available? Tyler: "Uh..." / Sasha: "We have nothing on this."
- Tyler embellishes personality. Sasha sticks to facts.
- Reference [LIVE CONTEXT] for medal counts or schedule data.
- Fun entertainment, not a textbook.
"""


def build_system_prompt(lang: str) -> str:
    lang_instr = I18N[lang].get("llm_lang_instruction", "Respond in English.")
    return (
        SYSTEM_PROMPT_BASE
        + f"\nLANGUAGE\n{lang_instr} "
        + "Keep character names Tyler and Sasha in English always.\n"
    )


# =========================================================
# 8. RAG RETRIEVAL
# =========================================================
def retrieve_context(query: str, top_k: int = 7) -> list:
    logger.info(f"Query: '{query}'")
    t0 = time.time()
    try:
        vec     = embedding_model.encode(query).tolist()
        results = pinecone_index.query(vector=vec, top_k=top_k, include_metadata=True)
        matches = results.get("matches", [])
        elapsed = time.time() - t0
        logger.info(f"Retrieved {len(matches)} chunks in {elapsed:.2f}s")
        for i, m in enumerate(matches):
            meta  = m.get("metadata", {})
            label = meta.get("name", meta.get("event", meta.get("moment", meta.get("storyline", ""))))
            logger.info(f"  [{i+1}] {meta.get('doc_type','?')} | {label} | score={m.get('score',0):.3f}")
        return matches
    except Exception as e:
        logger.error(f"Retrieval failed: {e}", exc_info=True)
        return []


def format_context_for_llm(matches: list, medal_df) -> str:
    parts = ["[RETRIEVED CONTEXT]"]
    for i, m in enumerate(matches, 1):
        meta  = m.get("metadata", {})
        text  = meta.get("text", "")
        dtype = meta.get("doc_type", "?")
        score = m.get("score", 0)
        parts.append(f"\n--- Chunk {i} (type={dtype}, relevance={score:.2f}) ---\n{text}")

    if medal_df is not None and not medal_df.empty:
        parts.append("\n\n[LIVE MEDAL STANDINGS — current]")
        parts.append(medal_df.head(15).to_string(index=False))

    return "\n".join(parts)


# =========================================================
# 9. GENERATION — Qwen2.5-7B-Instruct via HuggingFace / Together AI
#    Serverless inference. No GPU needed locally.
#    Free tier: ~few hundred req/hr. PRO ($9/mo): 20x more.
#    Multilingual (29 langs incl EN/FR/IT). Strong structured output.
# =========================================================
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def generate_response(user_query: str, context_text: str, lang: str) -> str:
    if hf_client is None:
        logger.error("generate_response called but hf_client is None — init must have failed.")
        return (
            "TYLER: Uh… something went wrong on our end.\n\n"
            "SASHA: The broadcast feed dropped. Try again."
        )
    logger.info(f"Calling {MODEL_ID} via HuggingFace…")
    t0 = time.time()
    try:
        messages = [
            {"role": "system", "content": build_system_prompt(lang)},
            {"role": "user",   "content": f"{context_text}\n\n[USER QUESTION]\n{user_query}"}
        ]

        output = hf_client.chat_completion(
            messages=messages,
            max_tokens=500,
            temperature=0.7,
            top_p=0.9
        )

        elapsed = time.time() - t0
        text = output.choices[0].message.content

        # token usage (available on most providers)
        usage = getattr(output, "usage", None)
        if usage:
            logger.info(
                f"Qwen responded in {elapsed:.2f}s | "
                f"in={usage.prompt_tokens} out={usage.completion_tokens} tokens"
            )
        else:
            logger.info(f"Qwen responded in {elapsed:.2f}s")

        return text

    except Exception as e:
        logger.error(f"HuggingFace inference error: {e}", exc_info=True)
        return (
            "TYLER: Uh… something went wrong on our end.\n\n"
            "SASHA: The broadcast feed dropped. Try again."
        )


# =========================================================
# 10. CSS
# =========================================================
CSS = """
<style>
/* ============================================================
 * MILANO CORTINA 2026 — OLYMPIC BRAND THEME
 *
 * Palette pulled from olympics.com Milano Cortina brand page:
 *   #0A1929  Deep Navy      — hero bg, primary text
 *   #00818A  Teal           — brand accent, gradient anchor
 *   #0033A0  Olympic Blue   — links, borders
 *   #006B3F  Forest Green   — Sasha accent
 *   #F4F7F8  Ice            — page background
 *   #FFFFFF  White          — cards, surfaces
 *   #E8ECEE  Frost          — dividers
 *   #6B7B8D  Slate          — captions, meta
 *
 * Tricolore: #009246 | #FFFFFF | #CE2B37
 * ============================================================ */

/* ── global ── */
/* Three layered ridgelines at low opacity — atmospheric mountain texture */
body, .stApp {
    background-color: #F4F7F8;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 400' preserveAspectRatio='xMidYMax slice'%3E%3Cpath fill='%230A1929' fill-opacity='0.018' d='M0,320 L120,260 L200,290 L340,200 L420,240 L520,180 L600,220 L720,150 L800,190 L920,120 L1000,170 L1100,100 L1200,160 L1300,130 L1440,180 L1440,400 L0,400 Z'/%3E%3Cpath fill='%230A1929' fill-opacity='0.025' d='M0,350 L80,310 L180,340 L280,280 L380,320 L460,260 L560,300 L680,240 L760,275 L860,220 L960,260 L1060,210 L1160,250 L1260,200 L1360,240 L1440,220 L1440,400 L0,400 Z'/%3E%3Cpath fill='%230A1929' fill-opacity='0.035' d='M0,370 L100,345 L200,365 L320,330 L400,355 L500,315 L600,350 L720,310 L820,340 L940,290 L1040,330 L1160,280 L1260,320 L1360,285 L1440,310 L1440,400 L0,400 Z'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: bottom center;
    background-size: 100% 420px;
    color: #0A1929;
    font-family: 'Georgia', serif;
}
.block-container {
    padding-top: 0 !important;
    padding-bottom: 2rem !important;
    max-width: 1320px !important;
}

/* ── hero header ── */
.header-band {
    background: linear-gradient(160deg, #0A1929 0%, #0D2540 55%, #112E4E 100%);
    padding: 2.6rem 1.2rem 2.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
/* subtle radial glow — gives depth like the brand page hero */
.header-band::before {
    content: '';
    position: absolute;
    top: -40%; left: 50%;
    transform: translateX(-50%);
    width: 140%; height: 140%;
    background: radial-gradient(ellipse at center, rgba(0,129,138,0.12) 0%, transparent 70%);
    pointer-events: none;
}
/* tricolore stripe at the very bottom */
.header-band::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg,
        #009246 0%, #009246 33.33%,
        #FFFFFF 33.33%, #FFFFFF 66.66%,
        #CE2B37 66.66%, #CE2B37 100%
    );
}
.header-band h1 {
    margin: 0;
    font-size: 2.3rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    position: relative;
}
/* "MILAN 2026" gets the teal-to-blue gradient text */
.header-band h1 .blue {
    background: linear-gradient(90deg, #00818A 0%, #40A8B5 50%, #5BB8C3 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.header-band .tagline {
    color: rgba(255,255,255,0.5);
    font-size: 0.88rem;
    margin-top: 0.5rem;
    font-style: italic;
    position: relative;
}

/* ── post-header breathing room ── */
.gap-below-header { height: 2.2rem; }

/* ── buttons base ── */
.stButton button {
    font-family: 'Segoe UI', system-ui, sans-serif !important;
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    min-height: 44px !important;
    padding: 0.5rem 0.7rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    background: #FFFFFF !important;
    border: 1.5px solid #E8ECEE !important;
    color: #0A1929 !important;
}
.stButton button:hover {
    border-color: #00818A !important;
    color: #00818A !important;
    background: #F0FAFA !important;
}
.stButton button:focus-visible {
    outline: 3px solid #00818A !important;
    outline-offset: 2px !important;
}

/* ── suggestion pills — teal gradient fill ── */
.try-label { display: block; }

[data-testid="column"] .try-label ~ * .stButton button,
[data-testid="stColumn"] .try-label ~ * .stButton button {
    background: linear-gradient(135deg, #00818A 0%, #0066B2 100%) !important;
    border: none !important;
    color: #FFFFFF !important;
    box-shadow: 0 3px 10px rgba(0,129,138,0.35) !important;
}
[data-testid="column"] .try-label ~ * .stButton button:hover,
[data-testid="stColumn"] .try-label ~ * .stButton button:hover {
    background: linear-gradient(135deg, #006B75 0%, #00508F 100%) !important;
    box-shadow: 0 4px 14px rgba(0,129,138,0.45) !important;
    transform: translateY(-1px) !important;
}

/* ── "Try asking" label ── */
.try-label {
    color: #6B7B8D;
    font-size: 0.7rem;
    font-weight: 600;
    font-family: 'Segoe UI', system-ui, sans-serif;
    margin-bottom: 0.45rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── chat bubbles ── */
.bubble {
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
    line-height: 1.6;
    animation: fadeUp 0.25s ease;
    color: #0A1929;
    font-size: 0.9rem;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(5px); }
    to   { opacity: 1; transform: translateY(0); }
}
.bubble-tyler {
    background: #EEF5FF;
    border-left: 4px solid #0033A0;
}
.bubble-sasha {
    background: #EDF7F1;
    border-left: 4px solid #006B3F;
}
.bubble .speaker {
    font-weight: 700;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.22rem;
    font-family: 'Segoe UI', system-ui, sans-serif;
}
.bubble-tyler .speaker { color: #0033A0; }
.bubble-sasha .speaker { color: #006B3F; }

/* ── user bubble ── */
.user-bubble {
    background: #FFFFFF;
    border-radius: 8px;
    border-right: 3px solid #00818A;
    padding: 0.5rem 0.85rem;
    margin-bottom: 0.2rem;
    text-align: right;
    color: #0A1929;
    font-size: 0.87rem;
    font-family: 'Segoe UI', system-ui, sans-serif;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}
.user-meta {
    color: #6B7B8D;
    font-size: 0.67rem;
    text-align: right;
    margin-bottom: 0.14rem;
    font-family: 'Segoe UI', system-ui, sans-serif;
}

/* ── turn divider ── */
.turn-divider {
    border: none;
    border-top: 1px solid #E8ECEE;
    margin: 1rem 0;
}

/* ── selectbox ── */
.stSelectbox [data-baseweb="select"] {
    font-family: 'Segoe UI', system-ui, sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #0A1929 !important;
    background: #FFFFFF !important;
    border: 1.5px solid #E8ECEE !important;
    border-radius: 6px !important;
    min-height: 38px !important;
    cursor: pointer !important;
}
.stSelectbox [data-baseweb="select"]:hover { border-color: #00818A !important; }
.stSelectbox [data-baseweb="select"]:focus-within {
    border-color: #00818A !important;
    box-shadow: 0 0 0 2px rgba(0,129,138,0.18) !important;
}
[data-baseweb="menu"] li {
    font-family: 'Segoe UI', system-ui, sans-serif !important;
    font-size: 0.78rem !important;
    color: #0A1929 !important;
}
[data-baseweb="menu"] li:hover { background: #F0FAFA !important; }

/* ── text input ── */
.stTextInput input {
    background: #FFFFFF !important;
    border: 1.5px solid #E8ECEE !important;
    color: #0A1929 !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
    min-height: 46px !important;
    padding: 0 0.8rem !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}
.stTextInput input:focus {
    border-color: #00818A !important;
    box-shadow: 0 0 0 3px rgba(0,129,138,0.18) !important;
    outline: none !important;
}
.stTextInput input::placeholder { color: #6B7B8D !important; }
.stTextInput label {
    color: #6B7B8D !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* ── countdown / competition day box ── */
.info-day-box {
    background: linear-gradient(145deg, #0A1929 0%, #00818A 100%);
    border-radius: 12px;
    padding: 1.6rem 1rem 1.4rem;
    margin-bottom: 0;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 18px rgba(0,129,138,0.25);
}
/* subtle inner glow */
.info-day-box::before {
    content: '';
    position: absolute;
    top: -30%; right: -20%;
    width: 70%; height: 70%;
    background: radial-gradient(circle, rgba(255,255,255,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.info-day-box .info-day-label {
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: rgba(255,255,255,0.6);
    margin-bottom: 0.35rem;
    position: relative;
}
.info-day-box .info-day-num {
    font-family: 'Georgia', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1;
    position: relative;
}
.info-day-box .info-day-date {
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 0.78rem;
    color: rgba(255,255,255,0.7);
    margin-top: 0.3rem;
    position: relative;
}

/* ── info panel section gap ── */
.info-section-gap { height: 1.4rem; }

/* ── section headings ── */
.sidebar-heading {
    color: #0A1929;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'Segoe UI', system-ui, sans-serif;
    border-bottom: 2px solid #00818A;
    padding-bottom: 0.3rem;
    margin-bottom: 0.6rem;
}

/* ── medal table ── */
.medal-table {
    width: 100%;
    border-collapse: collapse;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #E8ECEE;
    font-family: 'Segoe UI', system-ui, sans-serif;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.medal-table thead tr {
    background: linear-gradient(135deg, #0A1929, #112E4E);
}
.medal-th {
    color: #FFFFFF;
    font-size: 0.63rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0.55rem 0.35rem;
    text-align: center;
    border-right: 1px solid rgba(255,255,255,0.1);
}
.medal-th:last-child { border-right: none; }
.medal-th-country { text-align: left; padding-left: 0.55rem; color: rgba(255,255,255,0.85); }
.medal-th-gold   { color: #F0C040; }
.medal-th-silver { color: #C8D0D6; }
.medal-th-bronze { color: #CD7F32; }
.medal-th-total  { color: #FFFFFF; }

.medal-table tbody tr {
    background: #FFFFFF;
    border-bottom: 1px solid #EEF1F2;
    transition: background 0.15s;
}
.medal-table tbody tr:hover { background: #F0FAFA; }
.medal-table tbody tr:last-child { border-bottom: none; }
.medal-table tbody tr:nth-child(even) { background: #F8FAFB; }
.medal-table tbody tr:nth-child(even):hover { background: #F0FAFA; }

.medal-country {
    font-size: 0.78rem;
    font-weight: 600;
    color: #0A1929;
    padding: 0.5rem 0.55rem;
    border-right: 1px solid #EEF1F2;
}
.medal-num {
    font-size: 0.82rem;
    font-weight: 700;
    text-align: center;
    color: #0A1929;
    padding: 0.5rem 0.35rem;
    border-right: 1px solid #EEF1F2;
}
.medal-num:last-child { border-right: none; }
.medal-total { color: #00818A; }

/* ── stat cards ── */
.stat-row {
    display: flex;
    gap: 0.7rem;
    margin-top: 0.1rem;
}
.stat-row .stat-card { flex: 1; margin-bottom: 0 !important; }
.stat-card {
    background: #FFFFFF;
    border: 1.5px solid #E8ECEE;
    border-radius: 10px;
    padding: 0.7rem 0.4rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.stat-card .stat-val {
    font-size: 1.5rem;
    font-weight: 700;
    color: #00818A;
    font-family: 'Georgia', serif;
}
.stat-card .stat-label {
    font-size: 0.6rem;
    color: #6B7B8D;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'Segoe UI', system-ui, sans-serif;
    margin-top: 0.12rem;
}

/* ── about text ── */
.about-block {
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 0.82rem;
    color: #0A1929;
    line-height: 1.55;
}
.about-block .about-name { font-weight: 700; }
.about-block .about-flag { font-size: 0.7rem; color: #6B7B8D; }
.about-block .about-divider { color: #E8ECEE; margin: 0.5rem 0; }
.about-block .about-stack { color: #6B7B8D; font-size: 0.76rem; margin-top: 0.4rem; }
.about-block .about-stack strong { color: #0A1929; }

/* ── conversation log ── */
.conv-log {
    background: #0A1929;
    border-radius: 10px;
    padding: 0.8rem 0.9rem;
    margin-top: 0.3rem;
    max-height: 240px;
    overflow-y: auto;
    font-family: 'Consolas', 'SF Mono', monospace;
    font-size: 0.68rem;
    line-height: 1.7;
    color: rgba(255,255,255,0.75);
}
.conv-log .log-time {
    color: #00818A;
    margin-right: 0.4rem;
}
.conv-log .log-query {
    color: #FFFFFF;
    font-weight: 600;
}
.conv-log .log-speaker-tyler { color: #5BA3E8; }
.conv-log .log-speaker-sasha { color: #5BD49A; }
.conv-log .log-line { margin-bottom: 0.18rem; }

/* ── dividers ── */
hr { border: none; border-top: 1px solid #E8ECEE !important; margin: 0.8rem 0 !important; }

/* ── scrollbar ── */
::-webkit-scrollbar       { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #D0D8DE; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00818A; }

/* ── Streamlit spinner tint ── */
.stSpinner > div { border-color: #00818A !important; }
</style>
"""


# =========================================================
# 11. RENDER HELPERS
# =========================================================
def render_bubbles(response_text: str):
    """
    Parses model output into (speaker, body) pairs, enforces strict
    Tyler/Sasha alternation (flips any consecutive duplicate), then
    renders each as a styled chat bubble.

    Handles two output formats:
      Format A (ideal):  "TYLER: some dialogue here"
      Format B (actual): flag emoji + name on its own line, dialogue on next
    """
    # ── Step 1: parse into (speaker, body) tuples ──
    lines = [l.strip() for l in response_text.split("\n")]
    parsed = []                            # list of ("tyler"|"sasha", "body text")
    current_speaker = None
    current_body    = []

    def commit():
        if current_speaker and current_body:
            parsed.append((current_speaker, " ".join(current_body)))

    for line in lines:
        if not line:
            continue
        upper = line.upper()
        is_tyler = upper.startswith("TYLER") or line.startswith("🇺🇸")
        is_sasha = upper.startswith("SASHA") or line.startswith("🇷🇺")

        if is_tyler or is_sasha:
            commit()
            current_speaker = "tyler" if is_tyler else "sasha"
            current_body = []
            if ":" in line:
                remainder = line.split(":", 1)[-1].strip()
                if remainder:
                    current_body.append(remainder)
        else:
            if current_speaker:
                current_body.append(line)

    commit()

    # ── Step 2: enforce alternation ──
    # Flip any consecutive duplicate speaker to the other one
    for i in range(1, len(parsed)):
        if parsed[i][0] == parsed[i - 1][0]:
            parsed[i] = ("sasha" if parsed[i][0] == "tyler" else "tyler", parsed[i][1])

    # ── Step 3: render ──
    for speaker, body in parsed:
        if speaker == "tyler":
            st.markdown(
                f'<div class="bubble bubble-tyler">'
                f'<div class="speaker">USA Tyler</div>{body}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="bubble bubble-sasha">'
                f'<div class="speaker">RUS Sasha</div>{body}</div>',
                unsafe_allow_html=True
            )


# =========================================================
# 12. MAIN
# =========================================================
def main():
    st.markdown(CSS, unsafe_allow_html=True)

    # ── session init ──
    if "lang" not in st.session_state:
        st.session_state["lang"] = "EN"
    if "input_gen" not in st.session_state:
        st.session_state["input_gen"] = 0
    if "history" not in st.session_state:
        st.session_state["history"] = []

    active_lang = st.session_state["lang"]

    # ── dates ──
    today        = datetime.now()
    games_start  = datetime(2026, 2, 6)
    games_end    = datetime(2026, 2, 22, 23, 59)
    during_games = games_start <= today <= games_end

    # ── hero header ──
    title_html = t("header_title").replace("MILAN 2026", '<span class="blue">MILAN 2026</span>')
    st.markdown(
        f'<div class="header-band">'
        f'<h1>{title_html}</h1>'
        f'<div class="tagline">{t("header_tagline")}</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    # breathing room below tricolore stripe
    st.markdown('<div class="gap-below-header"></div>', unsafe_allow_html=True)

    # ── live data ──
    medal_df, medal_time, medal_err = fetch_live_medals()

    # ── two-column layout ──
    chat_col, info_col = st.columns([1.7, 1], gap="large")

    # ═══════════════════════════════════════════════
    # COLUMN 1 — CONVERSATION
    # ═══════════════════════════════════════════════
    with chat_col:
        # suggestion pills
        static_pills = t("suggestions_static")
        pills = [(s, s) for s in static_pills]
        if during_games:
            pills.insert(2, (t("suggestion_schedule"),
                             t("suggestion_schedule_query").format(date=today.strftime("%B %d"))))
        else:
            pills.insert(2, (t("suggestion_schedule_off"),
                             t("suggestion_schedule_off")))

        st.markdown(
            f'<p class="try-label">{t("try_asking")}</p>',
            unsafe_allow_html=True
        )
        pill_cols = st.columns(len(pills), gap="small")
        for col, (label, query_text) in zip(pill_cols, pills):
            if col.button(label, use_container_width=True, key=f"pill_{hash(label)}_{active_lang}"):
                st.session_state["pending_query"] = query_text
                st.rerun()

        pending = st.session_state.pop("pending_query", "")

        input_key = f"main_input_{st.session_state['input_gen']}"
        typed = st.text_input(
            t("input_label"),
            placeholder=t("input_placeholder"),
            key=input_key,
            value=pending,
            max_chars=300
        )

        query = pending if pending else typed

        if query and query.strip():
            log_and_show("info", f"Query [{active_lang}]: {query}")
            with st.spinner(t("spinner_text")):
                matches      = retrieve_context(query, top_k=7)
                log_and_show("info", f"Retrieved {len(matches)} chunks")
                context_text = format_context_for_llm(matches, medal_df)
                response     = generate_response(query, context_text, active_lang)
                log_and_show("info", "Response generated.")

            st.session_state["history"].append({
                "query":    query,
                "response": response,
                "time":     datetime.now().strftime("%I:%M %p"),
                "chunks":   len(matches),
                "lang":     active_lang
            })

            if pending:
                st.session_state["input_gen"] += 1
                st.rerun()

        # chat history (newest first)
        for turn in reversed(st.session_state.get("history", [])):
            st.markdown(
                f'<div class="user-meta">🕐 {turn["time"]} · {turn["lang"]}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="user-bubble">{turn["query"]}</div>',
                unsafe_allow_html=True
            )
            render_bubbles(turn["response"])
            st.markdown('<hr class="turn-divider">', unsafe_allow_html=True)

        # ── conversation log (terminal-style replay) ──
        history = st.session_state.get("history", [])
        if history:
            log_html = '<div class="conv-log">'
            for turn in history:   # oldest first for a log feel
                log_html += (
                    f'<div class="log-line">'
                    f'<span class="log-time">🕐 {turn["time"]} · {turn["lang"]}</span>'
                    f'<span class="log-query">{turn["query"]}</span>'
                    f'</div>'
                )
                # replay each bubble as a log line
                lines = [l.strip() for l in turn["response"].split("\n") if l.strip()]
                current_speaker = None
                current_body = []

                def flush_log():
                    nonlocal log_html
                    if current_speaker and current_body:
                        cls = "log-speaker-tyler" if current_speaker == "tyler" else "log-speaker-sasha"
                        flag = "USA" if current_speaker == "tyler" else "RUS"
                        name = "Tyler" if current_speaker == "tyler" else "Sasha"
                        log_html += (
                            f'<div class="log-line">'
                            f'<span class="{cls}">{flag} {name}:</span> {" ".join(current_body)}'
                            f'</div>'
                        )

                for line in lines:
                    upper = line.upper()
                    is_tyler = upper.startswith("TYLER") or line.startswith("🇺🇸")
                    is_sasha = upper.startswith("SASHA") or line.startswith("🇷🇺")
                    if is_tyler or is_sasha:
                        flush_log()
                        current_speaker = "tyler" if is_tyler else "sasha"
                        current_body = []
                        if ":" in line:
                            remainder = line.split(":", 1)[-1].strip()
                            if remainder:
                                current_body.append(remainder)
                    else:
                        if current_speaker:
                            current_body.append(line)
                flush_log()

            log_html += '</div>'

            with st.expander("📋 Conversation Log", expanded=False):
                st.markdown(log_html, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════
    # COLUMN 2 — INFO PANEL
    # ═══════════════════════════════════════════════
    with info_col:

        # ── Competition Day / Countdown ──
        if during_games:
            day_num  = (today - games_start).days + 1
            date_str = today.strftime("%A, %B %d")
            st.markdown(
                f'<div class="info-day-box">'
                f'<div class="info-day-label">Competition Day</div>'
                f'<div class="info-day-num">Day {day_num}</div>'
                f'<div class="info-day-date">{date_str}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        elif today < games_start:
            countdown = (games_start - today).days
            st.markdown(
                f'<div class="info-day-box">'
                f'<div class="info-day-label">Milano Cortina 2026</div>'
                f'<div class="info-day-num">{countdown} days</div>'
                f'<div class="info-day-date">Until the Games begin</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="info-day-box">'
                f'<div class="info-day-label">Milano Cortina 2026</div>'
                f'<div class="info-day-num">Finished</div>'
                f'<div class="info-day-date">Feb 6 – Feb 22, 2026</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        # gap between countdown and medal table
        st.markdown('<div class="info-section-gap"></div>', unsafe_allow_html=True)

        # ── Medal Standings ──
        st.markdown(f'<div class="sidebar-heading">🏅 {t("standings_title")}</div>', unsafe_allow_html=True)

        if medal_df is not None and not medal_df.empty:
            col_map = {}
            for c in medal_df.columns:
                cl = str(c).lower().strip()
                if cl in ("nation", "country", "noc", "nations"): col_map[c] = "Country"
                elif cl == "gold":   col_map[c] = "Gold"
                elif cl == "silver": col_map[c] = "Silver"
                elif cl == "bronze": col_map[c] = "Bronze"
                elif cl == "total":  col_map[c] = "Total"
            medal_df = medal_df.rename(columns=col_map)

            keep = [c for c in ["Country", "Gold", "Silver", "Bronze", "Total"] if c in medal_df.columns]
            if "Country" in medal_df.columns:
                mask = medal_df["Country"].astype(str).apply(
                    lambda x: (
                        x.strip() != "" and
                        not x.strip()[0].isdigit() and
                        "total" not in x.lower() and
                        "neutral" not in x.lower() and
                        "ain" != x.strip().lower()
                    )
                )
                medal_df = medal_df.loc[mask].reset_index(drop=True)
            top3 = medal_df[keep].head(3).reset_index(drop=True)

            rows_html = ""
            for i, row in top3.iterrows():
                country = str(row.get("Country", "—"))
                gold   = str(int(row["Gold"])) if "Gold" in row else "—"
                silver = str(int(row["Silver"])) if "Silver" in row else "—"
                bronze = str(int(row["Bronze"])) if "Bronze" in row else "—"
                total  = str(int(row["Total"])) if "Total" in row else "—"
                rows_html += (
                    f'<tr>'
                    f'<td class="medal-country">{country}</td>'
                    f'<td class="medal-num">{gold}</td>'
                    f'<td class="medal-num">{silver}</td>'
                    f'<td class="medal-num">{bronze}</td>'
                    f'<td class="medal-num medal-total">{total}</td>'
                    f'</tr>'
                )

            table_html = (
                '<table class="medal-table">'
                '<thead><tr>'
                '<th class="medal-th medal-th-country">Country</th>'
                '<th class="medal-th medal-th-gold">🥇</th>'
                '<th class="medal-th medal-th-silver">🥈</th>'
                '<th class="medal-th medal-th-bronze">🥉</th>'
                '<th class="medal-th medal-th-total">Total</th>'
                '</tr></thead>'
                f'<tbody>{rows_html}</tbody>'
                '</table>'
            )
            st.markdown(table_html, unsafe_allow_html=True)
        else:
            st.caption(medal_err or t("games_not_started"))

        # gap
        st.markdown('<div class="info-section-gap"></div>', unsafe_allow_html=True)

        # ── Medals Awarded + Athletes Tracked ──
        total_medals = "—"
        if medal_df is not None and not medal_df.empty:
            for cn in ["Total", "total"]:
                if cn in medal_df.columns:
                    try:
                        total_medals = f"{medal_df[cn].sum():,}"
                    except Exception:
                        pass
                    break

        st.markdown(
            f'<div class="stat-row">'
            f'<div class="stat-card">'
            f'<div class="stat-val">{total_medals}</div>'
            f'<div class="stat-label">{t("medals_label")}</div></div>'
            f'<div class="stat-card">'
            f'<div class="stat-val">407</div>'
            f'<div class="stat-label">{t("athletes_label")}</div></div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # gap
        st.markdown('<div class="info-section-gap"></div>', unsafe_allow_html=True)

        # ── About ──
        st.markdown(f'<div class="sidebar-heading">{t("about_title")}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="about-block">'
            '<span class="about-name">Tyler</span> <span class="about-flag">USA — 2018 Bronze · Figure Skating</span><br>'
            '<span class="about-name">Sasha</span> <span class="about-flag">RUS — 2014 & 2018 Silver · Figure Skating</span>'
            '<div class="about-divider">─</div>'
            'Rivals 2014–2018. Now partners. It\'s complicated.'
            '<div class="about-stack"><strong>Stack:</strong> Pinecone · Sentence Transformers · Wikipedia</div>'
            '</div>',
            unsafe_allow_html=True
        )

        # gap
        st.markdown('<div class="info-section-gap"></div>', unsafe_allow_html=True)

        # ── Language ──
        st.markdown(f'<div class="sidebar-heading">{t("about_title") if False else "Language"}</div>', unsafe_allow_html=True)
        lang_options = {"EN": "🇬🇧 English", "FR": "🇫🇷 Français", "IT": "🇮🇹 Italiano"}
        selected = st.selectbox(
            "Language",
            options=list(lang_options.keys()),
            format_func=lambda k: lang_options[k],
            index=list(lang_options.keys()).index(active_lang),
            key="lang_select",
            label_visibility="collapsed"
        )
        if selected != active_lang:
            st.session_state["lang"] = selected
            st.rerun()


if __name__ == "__main__":
    main()
