"""
milan2026_nlu.py
================
Natural Language Understanding for Milano Cortina 2026 content.

Extracts structured information from RSS feeds and text:
- Entities: athletes, countries, sports, events
- Topics: what type of content (injury, upset, rivalry, etc.)
- Sentiment: positive/negative/neutral
- Quality score: filter clickbait and low-quality content

Uses lightweight NLU (no heavy models):
  - spaCy for NER (Named Entity Recognition)
  - Keyword matching for sports/events
  - Rule-based sentiment
  - Heuristic quality scoring
"""

import re
from typing import Dict, List, Set, Optional
from dataclasses import dataclass

# Optional: Use spaCy for better NER if available
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False
    nlp = None


# ═══════════════════════════════════════════════════════════
# ENTITY DICTIONARIES
# ═══════════════════════════════════════════════════════════

WINTER_SPORTS = {
    "alpine skiing", "figure skating", "ice hockey", "curling",
    "bobsled", "bobsleigh", "skeleton", "luge", "biathlon",
    "cross-country", "ski jumping", "snowboard", "snowboarding",
    "speed skating", "freestyle skiing", "nordic combined",
    "short track", "ski mountaineering"
}

KNOWN_ATHLETES = {
    # Figure Skating
    "ilia malinin", "yuma kagiyama", "shoma uno", "adam siao him fa",
    "gabriella papadakis", "guillaume cizeron", "madison chock", "evan bates",
    "riku miura", "ryuichi kihara", "alexa knierim", "brandon frazier",
    
    # Alpine Skiing
    "mikaela shiffrin", "petra vlhova", "sara hector", "lara gut-behrami",
    "federica brignone", "marco odermatt", "alexis pinturault",
    
    # Ice Hockey
    "hilary knight", "kendall coyne schofield", "lee stecklein",
    "connor mcdavid", "auston matthews", "sidney crosby",
    
    # Speed Skating
    "irene schouten", "nao kodaira", "erin jackson",
    
    # Snowboarding
    "chloe kim", "shaun white", "red gerard",
    
    # Cross-Country
    "jessie diggins", "therese johaug", "alexander bolshunov",
    
    # Curling
    "john shuster",
    
    # Add more as needed
}

COUNTRIES = {
    "usa", "united states", "canada", "norway", "germany", "austria",
    "switzerland", "france", "italy", "japan", "china", "russia", "roc",
    "netherlands", "sweden", "finland", "czech republic", "south korea",
    "great britain", "slovenia", "poland", "spain"
}

TOPIC_KEYWORDS = {
    "injury": ["injury", "injured", "hurt", "sprain", "fracture", "withdraw", "pulled out", "sidelined"],
    "upset": ["upset", "surprise", "shock", "unexpected", "underdog", "stunner", "unforeseen"],
    "rivalry": ["rivalry", "rival", "versus", "vs", "clash", "showdown", "battle", "face off"],
    "comeback": ["comeback", "return", "recovering", "back from", "triumphant return"],
    "record": ["record", "world record", "olympic record", "fastest", "first ever", "historic"],
    "gold_hunt": ["gold medal", "gold hunt", "chasing gold", "medal favorite", "podium contender"],
    "controversy": ["controversy", "dispute", "scandal", "protest", "appeal", "disqualified"],
    "opening_ceremony": ["opening ceremony", "ceremony", "torch", "parade of nations"],
    "closing_ceremony": ["closing ceremony", "finale", "celebration"],
}

SENTIMENT_POSITIVE = {
    "triumph", "victory", "champion", "gold", "success", "amazing", "incredible",
    "stellar", "dominant", "flawless", "perfect", "historic", "comeback", "breakthrough"
}

SENTIMENT_NEGATIVE = {
    "loss", "defeat", "failed", "disappointing", "injury", "withdraw", "crash",
    "fall", "stumble", "controversy", "scandal", "upset", "shock", "heartbreak"
}


# ═══════════════════════════════════════════════════════════
# NLU OUTPUT
# ═══════════════════════════════════════════════════════════

@dataclass
class NLUResult:
    """Structured NLU output."""
    # Entities
    athletes: Set[str]
    countries: Set[str]
    sports: Set[str]
    
    # Topics
    topics: Set[str]
    
    # Sentiment
    sentiment: str  # "positive", "negative", "neutral"
    sentiment_score: float  # -1.0 to 1.0
    
    # Quality
    quality_score: float  # 0.0 to 1.0
    is_high_quality: bool
    
    # Metadata
    key_phrases: List[str]


# ═══════════════════════════════════════════════════════════
# ENTITY EXTRACTION
# ═══════════════════════════════════════════════════════════

def extract_entities(text: str) -> Dict[str, Set[str]]:
    """
    Extract athletes, countries, and sports from text.
    Uses both spaCy NER (if available) and keyword matching.
    """
    text_lower = text.lower()
    
    athletes = set()
    countries = set()
    sports = set()
    
    # Keyword matching for sports (always reliable)
    for sport in WINTER_SPORTS:
        if sport in text_lower:
            sports.add(sport)
    
    # Keyword matching for known athletes
    for athlete in KNOWN_ATHLETES:
        if athlete in text_lower:
            athletes.add(athlete.title())
    
    # Keyword matching for countries
    for country in COUNTRIES:
        if country in text_lower:
            countries.add(country.upper() if len(country) <= 3 else country.title())
    
    # If spaCy available, use it for additional NER
    if SPACY_AVAILABLE and nlp:
        doc = nlp(text[:1000])  # Limit to first 1000 chars for speed
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Check if it's likely an athlete (mentioned near sports keywords)
                context = text_lower[max(0, ent.start_char-100):min(len(text), ent.end_char+100)]
                if any(sport in context for sport in WINTER_SPORTS):
                    athletes.add(ent.text)
            
            elif ent.label_ == "GPE":  # Geopolitical entity
                countries.add(ent.text)
    
    return {
        "athletes": athletes,
        "countries": countries,
        "sports": sports
    }


# ═══════════════════════════════════════════════════════════
# TOPIC CLASSIFICATION
# ═══════════════════════════════════════════════════════════

def classify_topics(text: str) -> Set[str]:
    """Identify what topics the content covers."""
    text_lower = text.lower()
    topics = set()
    
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            topics.add(topic)
    
    # If no topics detected, classify as "general"
    if not topics:
        topics.add("general")
    
    return topics


# ═══════════════════════════════════════════════════════════
# SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════════

def analyze_sentiment(text: str) -> tuple[str, float]:
    """
    Simple rule-based sentiment analysis.
    Returns (label, score) where:
      label: "positive", "negative", "neutral"
      score: -1.0 (very negative) to 1.0 (very positive)
    """
    text_lower = text.lower()
    
    pos_count = sum(1 for word in SENTIMENT_POSITIVE if word in text_lower)
    neg_count = sum(1 for word in SENTIMENT_NEGATIVE if word in text_lower)
    
    # Calculate score
    total = pos_count + neg_count
    if total == 0:
        return "neutral", 0.0
    
    score = (pos_count - neg_count) / max(total, 1)
    
    # Classify
    if score > 0.2:
        label = "positive"
    elif score < -0.2:
        label = "negative"
    else:
        label = "neutral"
    
    return label, score


# ═══════════════════════════════════════════════════════════
# QUALITY SCORING
# ═══════════════════════════════════════════════════════════

def score_quality(text: str, title: str = "") -> float:
    """
    Heuristic quality scoring to filter clickbait and low-quality content.
    Returns 0.0 (low quality) to 1.0 (high quality).
    """
    score = 0.5  # Start at neutral
    
    # Length checks
    word_count = len(text.split())
    if word_count < 30:
        score -= 0.2  # Too short
    elif 50 <= word_count <= 500:
        score += 0.1  # Good length
    
    # Clickbait detection (title)
    clickbait_patterns = [
        r"\d+ (things|reasons|ways|secrets|tricks)",
        r"you won't believe",
        r"shocking",
        r"will blow your mind",
        r"number \d+ will",
    ]
    if title and any(re.search(pattern, title.lower()) for pattern in clickbait_patterns):
        score -= 0.3
    
    # Check for substantive content
    if any(word in text.lower() for word in ["according to", "reported", "announced", "confirmed"]):
        score += 0.1  # Has sources
    
    # Check for Olympic-specific content
    olympic_terms = ["olympic", "medal", "gold", "silver", "bronze", "games", "competition"]
    if sum(1 for term in olympic_terms if term in text.lower()) >= 2:
        score += 0.1
    
    # Check for athlete/country mentions
    entities = extract_entities(text)
    if len(entities["athletes"]) > 0:
        score += 0.1
    if len(entities["countries"]) > 0:
        score += 0.05
    
    # Clamp to 0-1
    return max(0.0, min(1.0, score))


# ═══════════════════════════════════════════════════════════
# KEY PHRASE EXTRACTION
# ═══════════════════════════════════════════════════════════

def extract_key_phrases(text: str, max_phrases: int = 5) -> List[str]:
    """Extract important phrases using simple heuristics."""
    sentences = re.split(r'[.!?]+', text)
    
    key_phrases = []
    
    for sentence in sentences[:10]:  # Check first 10 sentences
        s = sentence.strip()
        if not s:
            continue
        
        # Look for quoted text
        quotes = re.findall(r'"([^"]+)"', s)
        key_phrases.extend(quotes[:2])
        
        # Look for sentences with important keywords
        if any(word in s.lower() for word in ["gold", "record", "first", "historic", "upset", "champion"]):
            if len(s.split()) <= 20:  # Not too long
                key_phrases.append(s)
        
        if len(key_phrases) >= max_phrases:
            break
    
    return key_phrases[:max_phrases]


# ═══════════════════════════════════════════════════════════
# MAIN NLU FUNCTION
# ═══════════════════════════════════════════════════════════

def analyze_content(text: str, title: str = "", min_quality: float = 0.3) -> Optional[NLUResult]:
    """
    Full NLU analysis of Olympic content.
    Returns None if quality is below threshold.
    """
    # Extract entities
    entities = extract_entities(text)
    
    # Classify topics
    topics = classify_topics(text)
    
    # Analyze sentiment
    sentiment_label, sentiment_score = analyze_sentiment(text)
    
    # Score quality
    quality = score_quality(text, title)
    
    # Filter low quality
    if quality < min_quality:
        return None
    
    # Extract key phrases
    key_phrases = extract_key_phrases(text)
    
    return NLUResult(
        athletes=entities["athletes"],
        countries=entities["countries"],
        sports=entities["sports"],
        topics=topics,
        sentiment=sentiment_label,
        sentiment_score=sentiment_score,
        quality_score=quality,
        is_high_quality=(quality >= 0.6),
        key_phrases=key_phrases
    )


# ═══════════════════════════════════════════════════════════
# HELPER: Enrich metadata
# ═══════════════════════════════════════════════════════════

def enrich_metadata(nlu_result: NLUResult) -> dict:
    """Convert NLU result to metadata dict for Pinecone."""
    return {
        "athletes": list(nlu_result.athletes),
        "countries": list(nlu_result.countries),
        "sports": list(nlu_result.sports),
        "topics": list(nlu_result.topics),
        "sentiment": nlu_result.sentiment,
        "sentiment_score": nlu_result.sentiment_score,
        "quality_score": nlu_result.quality_score,
        "is_high_quality": nlu_result.is_high_quality,
        "key_phrases": nlu_result.key_phrases[:3],  # Limit to 3 for Pinecone
    }
