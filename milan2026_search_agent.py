# How to Know What's Being Added to Your RSS Pipeline

## **Three Ways to Monitor Your RSS Agent**

---

## **Method 1: Check GitHub Actions Logs** (Easiest)

After each RSS agent run, go to:
```
Your Repo → Actions Tab → Click latest workflow run → Click "rss-agent" job
```

### **You'll See This Summary:**

```
======================================================================
FEED SUMMARY
======================================================================
  Olympic Channel          │ Raw: 201 │ Kept: 120 │ Filtered:  81
  BBC Sport Olympics       │ Raw:   4 │ Kept:   4 │ Filtered:   0
  Team USA                 │ Raw:  20 │ Kept:  15 │ Filtered:   5
  AP News Olympics         │ Raw:  25 │ Kept:  25 │ Filtered:   0
======================================================================
  TOTALS: Raw: 250, Kept: 164, Filtered: 86
======================================================================
```

**What This Tells You:**

| Column | Meaning |
|--------|---------|
| **Raw** | Total articles fetched from the feed |
| **Kept** | Articles that passed the Olympic filter |
| **Filtered** | Articles removed (non-Olympic content) |

**Key Insights:**
- **AP News Olympics: 0 Filtered** → AP hub is working! All content is relevant
- **Olympic Channel: 81 Filtered** → This feed has lots of non-winter sports content
- **Totals** → Overall pipeline health

---

## **Method 2: Enable DEBUG Mode** (See Every Article)

### **Step 1: Update Workflow**

In `.github/workflows/rss_agent_30min.yml`, change:

```yaml
# FROM:
- name: Run RSS search agent
  env:
    PINECONE_API_KEY: ${{ secrets.PINECONE_API_KEY }}
    AGENT_LOG_LEVEL: INFO    # ← Change this

# TO:
- name: Run RSS search agent
  env:
    PINECONE_API_KEY: ${{ secrets.PINECONE_API_KEY }}
    AGENT_LOG_LEVEL: DEBUG   # ← More verbose!
```

### **Step 2: Check Logs**

You'll now see every article:

```
[DEBUG] KEPT: Vonn confident she can race at Olympics
  Source: ap_olympics (trusted hub)
  
[DEBUG] KEPT: Speedskater Erin Jackson picked as flagbearer
  Matched keyword: bobsledder
  
[DEBUG] FILTERED (no Olympic keywords): NBA Finals Game 5 recap
  URL: https://olympic_channel.com/nba-finals
  
[DEBUG] NLU enriched: 2 athletes, 1 topics
```

**What This Shows:**
- ✅ **KEPT** → Article was added to Pinecone
- ❌ **FILTERED** → Article was removed (with reason)
- **Matched keyword** → Which keyword triggered the keep
- **NLU enriched** → How many athletes/topics extracted

---

## **Method 3: Query Pinecone Directly** (See What's Stored)

### **Option A: Use Pinecone Console**

1. Go to https://app.pinecone.io
2. Click your index: `milan-2026-olympics`
3. Click "Fetch" tab
4. Namespace: `narratives`
5. See all stored vectors with metadata

### **Option B: Query via API**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-key")
index = pc.Index("milan-2026-olympics")

# Get stats
stats = index.describe_index_stats()
print(f"Narratives namespace: {stats['namespaces']['narratives']['vector_count']} vectors")

# Fetch recent vectors
results = index.query(
    vector=[0.1] * 384,  # Dummy vector
    top_k=10,
    namespace="narratives",
    include_metadata=True
)

for match in results['matches']:
    print(f"\n{match['metadata']['source_key']}")
    print(f"  {match['metadata']['text'][:100]}...")
    print(f"  URL: {match['metadata']['url']}")
```

---

## **Method 4: Add Article Titles to Logs** (Custom)

Want to see article titles in the summary? Add this to your search agent:

```python
def log_feed_summary_with_titles(all_chunks: list[dict], olympic_chunks: list[dict]):
    """Enhanced summary with sample article titles"""
    from collections import Counter, defaultdict
    
    logger.info("=" * 70)
    logger.info("FEED SUMMARY (with sample articles)")
    logger.info("=" * 70)
    
    # Group by source
    raw_counts = Counter(c["source_key"] for c in all_chunks)
    filtered_counts = Counter(c["source_key"] for c in olympic_chunks)
    
    # Collect kept articles per feed
    kept_by_feed = defaultdict(list)
    for chunk in olympic_chunks:
        kept_by_feed[chunk["source_key"]].append(chunk.get("title", chunk["text"][:50]))
    
    for feed_key in sorted(raw_counts.keys()):
        raw = raw_counts[feed_key]
        filtered = filtered_counts.get(feed_key, 0)
        removed = raw - filtered
        
        logger.info(f"\n{feed_key:20} | Raw: {raw:3} | Kept: {filtered:3} | Filtered: {removed:3}")
        
        # Show sample titles
        if kept_by_feed[feed_key]:
            logger.info(f"  Sample articles:")
            for title in kept_by_feed[feed_key][:3]:  # Show first 3
                logger.info(f"    • {title[:60]}...")
```

---

## **Real Example from Your Logs**

### **Current State:**
```
[INFO] Fetching RSS: AP News Olympics …
[INFO]   → 0 chunks from AP News Olympics
```
**Problem:** Filter too strict

### **After Fix:**
```
[INFO] Fetching RSS: AP News Olympics …
[INFO]   → 25 chunks from AP News Olympics

======================================================================
FEED SUMMARY
======================================================================
  AP News Olympics         │ Raw:  25 │ Kept:  25 │ Filtered:   0
======================================================================
```
**Success:** All AP Milano Cortina articles captured!

---

## **Quick Reference: What Each Log Level Shows**

| Level | What You See |
|-------|--------------|
| **INFO** | Feed counts, totals, summary table |
| **DEBUG** | Every article title, filter decisions, NLU details |
| **WARNING** | Only errors (RSS parse failures, API errors) |

---

## **Best Practice: Progressive Monitoring**

### **Start with INFO** (default)
- Quick overview
- See totals and feed health
- 5-10 lines per run

### **Switch to DEBUG when troubleshooting**
- See individual articles
- Understand filter decisions
- 100+ lines per run

### **Back to INFO once working**
- Keep logs clean
- Summary table is enough

---

## **TL;DR - Quick Check**

After deploying the new agent, look for this in your GitHub Actions logs:

```
======================================================================
FEED SUMMARY
======================================================================
  AP News Olympics         │ Raw:  25 │ Kept:  25 │ Filtered:   0  ← Should be 20-30
======================================================================
```

If **Kept > 0**, your AP News feed is working! 🎉
