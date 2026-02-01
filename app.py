"""
MILAN 2026 WINTER OLYMPICS — TYLER & SASHA
============================================
Production Streamlit app. Deploy via Streamlit Community Cloud from GitHub.

Requirements: see requirements.txt
Secrets: PINECONE_API_KEY, HF_TOKEN  (in .streamlit/secrets.toml)

Architecture:
  Pinecone              -> semantic search (athletes / history / storylines / schedule)
  SentenceTransformers  -> FREE local embeddings (all-MiniLM-L6-v2, 384-dim)
  HuggingFace Inference -> Qwen2.5-3B-Instruct, serverless, no GPU needed
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
        "about_text":        "**Tyler** 🇺🇸 — 2018 Bronze · Figure Skating\n**Sasha** 🇷🇺 — 2014 & 2018 Silver · Figure Skating\n\nRivals 2014–2018. Now partners. It's complicated.\n\n**Stack:** Pinecone · Sentence Transformers · Haiku · Wikipedia",
        "games_not_started": "Medal table not yet available. Games start Feb 6.",
        "suggestions": [
            "Who should I watch in figure skating?",
            "Who are the USA medal favorites?",
            "What's on the schedule for Feb 11?",
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
        "about_text":        "**Tyler** 🇺🇸 — Bronze 2018 · Patinage artistique\n**Sasha** 🇷🇺 — Argent 2014 & 2018 · Patinage artistique\n\nRivaux 2014–2018. Maintenant partenaires. C'est compliqué.\n\n**Pile :** Pinecone · Sentence Transformers · Haiku · Wikipedia",
        "games_not_started": "Le tableau des médailles n'est pas encore disponible. Les Jeux commencent le 6 février.",
        "suggestions": [
            "Qui regarder en patinage artistique ?",
            "Qui sont les favorites pour une médaille (USA) ?",
            "Qu'est-il prévu pour le 11 février ?",
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
        "about_text":        "**Tyler** 🇺🇸 — Bronzo 2018 · Pattinaggio artistico\n**Sasha** 🇷🇺 — Argento 2014 & 2018 · Pattinaggio artistico\n\nRivali 2014–2018. Ora partner. È complicato.\n\n**Stack:** Pinecone · Sentence Transformers · Haiku · Wikipedia",
        "games_not_started": "La tabella delle medaglie non è ancora disponibile. I Giochi iniziano il 6 febbraio.",
        "suggestions": [
            "Chi guardare nel pattinaggio artistico?",
            "Chi sono i favoriti per la medaglia (USA)?",
            "Cosa è previsto per il 11 febbraio?",
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
    client = InferenceClient(
        model="Qwen/Qwen2.5-3B-Instruct",
        token=HF_TOKEN,
        provider="together"
    )
    logger.info("HuggingFace InferenceClient ready (Qwen2.5-3B-Instruct via Together AI).")
    return client


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
Always back-and-forth. Tyler first. Exactly like this:

TYLER: [line]

SASHA: [line]

TYLER: [optional]

SASHA: [optional]

2-4 exchanges max. Conversational. Let personality do the work.

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
    t0      = time.time()
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
# 9. GENERATION — Qwen2.5-3B-Instruct via HuggingFace
#    Serverless inference. No GPU needed locally.
#    Free tier: ~few hundred req/hr. PRO ($9/mo): 20x more.
#    Multilingual (29 langs incl EN/FR/IT). Strong structured output.
# =========================================================
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def generate_response(user_query: str, context_text: str, lang: str) -> str:
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
        logger.error(f"HuggingFace inference error: {e}")
        return (
            "TYLER: Uh… something went wrong on our end.\n\n"
            "SASHA: The broadcast feed dropped. Try again."
        )


# =========================================================
# 10. CSS
# =========================================================
CSS = """
<style>
body, .stApp {
    background: #0d1b2a;
    color: #c9d6df;
    font-family: 'Segoe UI', system-ui, sans-serif;
}
.block-container { padding-top: 0.9rem; padding-bottom: 0.7rem; max-width: 1180px; }

/* header */
.header-band {
    background: linear-gradient(135deg, #0a1628 0%, #162a45 50%, #0a1628 100%);
    border-bottom: 2px solid #1e3a5f;
    padding: 1.1rem 1.4rem;
    text-align: center;
    position: relative; overflow: hidden;
}
.header-band::before {
    content: '';
    position: absolute; inset: 0;
    background: repeating-linear-gradient(
        90deg, transparent, transparent 58px,
        rgba(30,90,140,0.07) 58px, rgba(30,90,140,0.07) 59px
    );
    pointer-events: none;
}
.header-band h1 {
    margin: 0; font-size: 1.85rem; font-weight: 700;
    color: #fff; letter-spacing: 0.07em; position: relative;
}
.header-band .tagline {
    color: #6a9cc6; font-size: 0.85rem;
    margin-top: 0.18rem; font-style: italic; position: relative;
}

/* lang toggle */
.lang-row { display: flex; justify-content: center; gap: 0.45rem; padding: 0.45rem 0; }

/* bubbles */
.bubble {
    border-radius: 12px; padding: 0.8rem 0.95rem;
    margin-bottom: 0.5rem; line-height: 1.5;
    animation: fadeUp 0.22s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(5px); }
    to   { opacity: 1; transform: translateY(0); }
}
.bubble-tyler {
    background: linear-gradient(135deg, #162d4a, #1e3a5c);
    border-left: 3px solid #e8a838;
}
.bubble-sasha {
    background: linear-gradient(135deg, #2a1b30, #3a2545);
    border-left: 3px solid #c94040;
}
.bubble .speaker {
    font-weight: 700; font-size: 0.73rem;
    letter-spacing: 0.09em; text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.bubble-tyler .speaker { color: #e8a838; }
.bubble-sasha .speaker { color: #c94040; }

/* user bubble */
.user-bubble {
    background: #1a2d4a; border-radius: 8px;
    padding: 0.45rem 0.8rem; margin-bottom: 0.35rem;
    text-align: right; color: #b8cfe0; font-size: 0.86rem;
    border-right: 3px solid #3a7cc6;
}
.user-meta { color: #4a7fa5; font-size: 0.7rem; text-align: right; margin-bottom: 0.08rem; }

/* input */
.stTextInput input {
    background: #152535 !important;
    border: 1px solid #2a4a6b !important;
    color: #e8edf2 !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
}
.stTextInput label { color: #6a9cc6 !important; font-size: 0.8rem !important; }

/* stat cards */
.stat-card {
    background: #152535; border: 1px solid #1e3a5f;
    border-radius: 10px; padding: 0.55rem 0.7rem;
    margin-bottom: 0.45rem; text-align: center;
}
.stat-card .stat-val   { font-size: 1.45rem; font-weight: 700; color: #fff; }
.stat-card .stat-label { font-size: 0.66rem; color: #5a8aad; text-transform: uppercase; letter-spacing: 0.07em; }

/* log panel */
.log-panel {
    background: #0b1520; border: 1px solid #1a3045;
    border-radius: 8px; padding: 0.55rem;
    max-height: 190px; overflow-y: auto;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 0.66rem; color: #4a7a9a; line-height: 1.4;
}
.log-panel .log-err  { color: #d65555; }
.log-panel .log-warn { color: #c9963a; }

/* buttons override for pills + lang */
.stButton button {
    background: #152d46 !important;
    border: 1px solid #2a4f72 !important;
    color: #7aadcf !important;
    border-radius: 18px !important;
    font-size: 0.74rem !important;
    padding: 0.25rem 0.6rem !important;
    transition: background 0.18s, color 0.18s !important;
}
.stButton button:hover { background: #1e3f5e !important; color: #fff !important; }

hr { border-color: #1a3045 !important; }
</style>
"""


# =========================================================
# 11. RENDER HELPERS
# =========================================================
def render_bubbles(response_text: str):
    for line in response_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        upper = line.upper()
        if "TYLER" in upper or line.startswith("🇺🇸"):
            body = line.split(":", 1)[-1].strip() if ":" in line else line
            st.markdown(
                f'<div class="bubble bubble-tyler">'
                f'<div class="speaker">🇺🇸 Tyler</div>{body}</div>',
                unsafe_allow_html=True
            )
        elif "SASHA" in upper or line.startswith("🇷🇺"):
            body = line.split(":", 1)[-1].strip() if ":" in line else line
            st.markdown(
                f'<div class="bubble bubble-sasha">'
                f'<div class="speaker">🇷🇺 Sasha</div>{body}</div>',
                unsafe_allow_html=True
            )


# =========================================================
# 12. MAIN
# =========================================================
def main():
    st.markdown(CSS, unsafe_allow_html=True)

    # ── language toggle ──
    if "lang" not in st.session_state:
        st.session_state["lang"] = "EN"

    active_lang = st.session_state["lang"]
    lang_cols   = st.columns(5)   # EN | FR | IT | spacer | spacer

    for i, (code, label) in enumerate([("EN","🇬🇧 EN"), ("FR","🇫🇷 FR"), ("IT","🇮🇹 IT")]):
        if lang_cols[i].button(label, key=f"lang_{code}", use_container_width=True):
            st.session_state["lang"] = code
            st.rerun()

    # ── header ──
    st.markdown(
        f'<div class="header-band">'
        f'<h1>{t("header_title")}</h1>'
        f'<div class="tagline">{t("header_tagline")}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── live data ──
    medal_df, medal_time, medal_err = fetch_live_medals()
    vector_count = get_pinecone_vector_count()

    # ── layout ──
    main_col, side_col = st.columns([2.3, 1], gap="medium")

    # ─────────── MAIN ───────────
    with main_col:
        if "history" not in st.session_state:
            st.session_state["history"] = []

        # suggestion pills
        suggestions = t("suggestions")
        st.markdown(
            f"<p style='color:#5a8aad;font-size:0.76rem;margin-bottom:0.25rem;'>{t('try_asking')}</p>",
            unsafe_allow_html=True
        )
        pill_cols = st.columns(len(suggestions), gap="small")
        for col, sug in zip(pill_cols, suggestions):
            if col.button(sug, use_container_width=True, key=f"pill_{hash(sug)}_{active_lang}"):
                st.session_state["pending_query"] = sug

        # input
        query = st.text_input(
            t("input_label"),
            placeholder=t("input_placeholder"),
            key="main_input",
            value=st.session_state.pop("pending_query", ""),
            max_chars=300
        )

        # process
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

        # chat history (newest first)
        for turn in reversed(st.session_state.get("history", [])):
            st.markdown(
                f'<div class="user-meta">🕐 {turn["time"]} · {turn["chunks"]} chunks · {turn["lang"]}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="user-bubble">🗨️ {turn["query"]}</div>',
                unsafe_allow_html=True
            )
            render_bubbles(turn["response"])
            st.markdown("<hr/>", unsafe_allow_html=True)

    # ─────────── SIDEBAR ───────────
    with side_col:
        st.markdown(f"### 📊 {t('dashboard_title')}")

        vc = f"{vector_count:,}" if vector_count else "—"
        st.markdown(
            f'<div class="stat-card"><div class="stat-val">{vc}</div>'
            f'<div class="stat-label">{t("vectors_label")}</div></div>',
            unsafe_allow_html=True
        )

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
            f'<div class="stat-card"><div class="stat-val">{total_medals}</div>'
            f'<div class="stat-label">{t("medals_label")}</div></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="stat-card"><div class="stat-val">407</div>'
            f'<div class="stat-label">{t("athletes_label")}</div></div>',
            unsafe_allow_html=True
        )

        # medal table
        st.markdown("---")
        st.markdown(f"### 🏅 {t('standings_title')}")
        st.caption(t("fetched_at").format(time=medal_time))

        if medal_df is not None and not medal_df.empty:
            display = medal_df.head(12).reset_index(drop=True)
            display.index = display.index + 1
            st.dataframe(display, use_container_width=True, hide_index=False)
        else:
            st.info(medal_err or t("games_not_started"))

        # log panel
        st.markdown("---")
        st.markdown(f"### 🔧 {t('log_title')}")

        entries = st.session_state.get("log_entries", [])
        if entries:
            html = '<div class="log-panel">'
            for e in reversed(entries[-20:]):
                css = "log-err" if "[ERROR]" in e else ("log-warn" if "[WARNING]" in e else "")
                html += f'<div class="{css}">{e}</div>'
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.caption(t("log_empty"))

        # about
        st.markdown("---")
        st.markdown(f"### {t('about_title')}")
        st.markdown(t("about_text"))


if __name__ == "__main__":
    main()
