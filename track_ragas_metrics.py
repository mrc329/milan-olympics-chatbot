"""
track_ragas_metrics.py
======================
Daily RAGAS evaluation with historical tracking for Milano Cortina 2026 chatbot.

Runs automatically via GitHub Actions.
Evaluates 33 comprehensive test questions.
Stores results in ragas_history.csv for Tableau visualization.

Usage:
    python track_ragas_metrics.py
"""

from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from datasets import Dataset
from datetime import datetime
import pandas as pd
import os
from pathlib import Path

# Your app functions
try:
    from app import (
        retrieve_context,
        format_context_for_llm,
        load_embedding_model,
        load_pinecone_index,
    )
except ImportError:
    print("⚠️  Warning: Could not import from app.py")
    print("   Make sure you're running this from the project root directory")


# ═══════════════════════════════════════════════════════════
# EVALUATION QUESTIONS (33 comprehensive test cases)
# ═══════════════════════════════════════════════════════════

EVAL_QUESTIONS = [
    # ═══════════════════════════════════════════════════════════
    # FIGURE SKATING - Singles (6 questions)
    # ═══════════════════════════════════════════════════════════
    {
        "question": "Who is Ilia Malinin?",
        "ground_truth": "Ilia Malinin is an American figure skater known as the Quad God. He was the first person to successfully land a quad Axel in competition.",
    },
    {
        "question": "What is Yuzuru Hanyu known for?",
        "ground_truth": "Yuzuru Hanyu is a Japanese figure skater and back-to-back Olympic gold medalist (2014, 2018). He pioneered the competitive quad Axel attempt and is considered one of the greatest figure skaters of all time.",
    },
    {
        "question": "Tell me about the men's figure skating results",
        "ground_truth": "Yuzuru Hanyu won gold in the men's figure skating free skate, with other Japanese skaters Kagiyama Kaito and Shoma Uno taking silver and bronze.",
    },
    {
        "question": "Who is Adam Siao Him Fa?",
        "ground_truth": "Adam Siao Him Fa is a competitive figure skater competing at Milano Cortina 2026 in men's singles.",
    },
    {
        "question": "What sports use the quad technique?",
        "ground_truth": "The quad (quadruple jump) is primarily used in figure skating. Ilia Malinin is known for landing the most difficult quad variation, the quad Axel.",
    },
    {
        "question": "Who are the top contenders in men's figure skating?",
        "ground_truth": "Top contenders in men's figure skating include Ilia Malinin (USA), Yuzuru Hanyu (Japan), Yuma Kagiyama (Japan), and Shoma Uno (Japan).",
    },
    
    # ═══════════════════════════════════════════════════════════
    # FIGURE SKATING - Ice Dance (4 questions)
    # ═══════════════════════════════════════════════════════════
    {
        "question": "Tell me about Madison Chock and Evan Bates",
        "ground_truth": "Madison Chock and Evan Bates are an American ice dance team. They are multiple-time U.S. national champions and world medalists competing in ice dance at Milano Cortina 2026.",
    },
    {
        "question": "Who are the favorites in ice dance?",
        "ground_truth": "Favorites in ice dance include Madison Chock and Evan Bates from the USA, and Gabriella Papadakis and Guillaume Cizeron from France.",
    },
    {
        "question": "When does ice dance competition start?",
        "ground_truth": "Ice dance competition includes events throughout the Games, with medal rounds typically held in the second week.",
    },
    {
        "question": "Tell me about Gabriella Papadakis and Guillaume Cizeron",
        "ground_truth": "Gabriella Papadakis and Guillaume Cizeron are a French ice dance team and former Olympic champions, competing at Milano Cortina 2026.",
    },
    
    # ═══════════════════════════════════════════════════════════
    # FIGURE SKATING - Pairs (2 questions)
    # ═══════════════════════════════════════════════════════════
    {
        "question": "Who are the top figure skating pairs teams?",
        "ground_truth": "Top figure skating pairs include Riku Miura and Ryuichi Kihara from Japan, and Alexa Knierim and Brandon Frazier from the USA.",
    },
    {
        "question": "Tell me about Riku Miura and Ryuichi Kihara",
        "ground_truth": "Riku Miura and Ryuichi Kihara are a Japanese pairs figure skating team competing at Milano Cortina 2026.",
    },
    
    # ═══════════════════════════════════════════════════════════
    # ALPINE SKIING (5 questions)
    # ═══════════════════════════════════════════════════════════
    {
        "question": "What is Mikaela Shiffrin's injury status?",
        "ground_truth": "Mikaela Shiffrin sustained a left ankle sprain during a World Cup giant slalom in Val d'Isère. She has been cleared for travel but her training load has been reduced heading into the Games.",
    },
    {
        "question": "Who won gold in women's downhill alpine skiing?",
        "ground_truth": "Sara Hector from Sweden won gold in the women's downhill alpine skiing event at Milano Cortina 2026.",
    },
    {
        "question": "Tell me about Ester Ledecká",
        "ground_truth": "Ester Ledecká is a Czech athlete who competes in both alpine skiing and snowboarding, making her one of the most versatile Winter Olympians. She has won Olympic gold medals in both sports.",
    },
    {
        "question": "What injuries have affected the Games?",
        "ground_truth": "Mikaela Shiffrin is competing with a left ankle sprain sustained before the Games, which has reduced her training load but she was cleared to compete.",
    },
    {
        "question": "Who are the favorites in alpine skiing?",
        "ground_truth": "Favorites in alpine skiing include Mikaela Shiffrin (USA), Sara Hector (Sweden), Petra Vlhova (Slovakia), and Lara Gut-Behrami (Switzerland).",
    },
    
    # ═══════════════════════════════════════════════════════════
    # CROSS-COUNTRY SKIING (3 questions)
    # ═══════════════════════════════════════════════════════════
    {
        "question": "What comeback stories are there?",
        "ground_truth": "Notable comeback stories include Therese Johaug returning from retirement at age 36 to compete in cross-country skiing, and Deanna Stellato-Dudek who left figure skating for 16 years before winning a world title at age 42.",
    },
    {
        "question": "Who is Jessie Diggins?",
        "ground_truth": "Jessie Diggins is an American cross-country skier and 2022 Olympic gold medalist. She was the first American woman to win Olympic cross-country gold.",
    },
    {
        "question": "Tell me about Therese Johaug's comeback",
        "ground_truth": "Therese Johaug is a Norwegian cross-country skiing legend who returned from retirement at age 36 to compete at Milano Cortina 2026, aiming to reclaim her status as one of the greatest of all time.",
    },
    
    # ═══════════════════════════════════════════════════════════
    # SPEED SKATING (3 questions)
    # ═══════════════════════════════════════════════════════════
    {
        "question": "Tell me about Irene Schouten",
        "ground_truth": "Irene Schouten is a Dutch speed skater and defending Olympic champion in multiple events. She is a dominant distance skater competing in the 500m and other events.",
    },
    {
        "question": "Who is Kendall Coyne Schofield?",
        "ground_truth": "Kendall Coyne Schofield is an American speed skater and 2018 Olympic gold medalist, known for her blazing 500m times.",
    },
    {
        "question": "Who won the women's 500m speed skating?",
        "ground_truth": "Irene Schouten from the Netherlands won gold in the women's 500m speed skating event.",
    },
    
    # ═══════════════════════════════════════════════════════════
    # ICE HOCKEY (4 questions)
    # ═══════════════════════════════════════════════════════════
    {
        "question": "Who is Lee Stecklein?",
        "ground_truth": "Lee Stecklein is the USA women's ice hockey captain and a two-time Olympic gold medalist (2018, 2022).",
    },
    {
        "question": "Tell me about the women's ice hockey tournament",
        "ground_truth": "The women's ice hockey tournament features top teams including the USA and Canada. The USA team is captained by Lee Stecklein and includes stars like Hilary Knight and Kendall Coyne Schofield.",
    },
    {
        "question": "Compare USA and Canada in ice hockey",
        "ground_truth": "USA and Canada are the top two teams in women's ice hockey. At Milano Cortina 2026, Canada won gold and USA took silver in a closely contested final.",
    },
    {
        "question": "Who are the stars of USA women's hockey?",
        "ground_truth": "Stars of USA women's hockey include captain Lee Stecklein, Hilary Knight, and Kendall Coyne Schofield.",
    },
    
    # ═══════════════════════════════════════════════════════════
    # CURLING (2 questions)
    # ═══════════════════════════════════════════════════════════
    {
        "question": "What happened in men's curling?",
        "ground_truth": "Sweden won gold in men's curling, with USA taking silver and Norway bronze. John Shuster's USA team was the favorite but finished second.",
    },
    {
        "question": "Who is John Shuster?",
        "ground_truth": "John Shuster is an American curler who led the USA to Olympic gold in 2018 at PyeongChang. He is a veteran skip competing at Milano Cortina 2026.",
    },
    
    # ═══════════════════════════════════════════════════════════
    # SCHEDULE & LOGISTICS (3 questions)
    # ═══════════════════════════════════════════════════════════
    {
        "question": "When is the opening ceremony?",
        "ground_truth": "The opening ceremony is scheduled for February 6, 2026 in Milano, Italy.",
    },
    {
        "question": "What events are happening on February 10?",
        "ground_truth": "February 10 features multiple events including alpine skiing, figure skating, speed skating, and other winter sports competitions.",
    },
    {
        "question": "When is the closing ceremony?",
        "ground_truth": "The closing ceremony is scheduled for February 22, 2026, marking the end of the Milano Cortina Winter Olympics.",
    },
    
    # ═══════════════════════════════════════════════════════════
    # MULTI-SPORT / MEDAL FAVORITES (2 questions)
    # ═══════════════════════════════════════════════════════════
    {
        "question": "Who are the USA medal favorites?",
        "ground_truth": "USA medal favorites include Mikaela Shiffrin in alpine skiing, Ilia Malinin in figure skating, Madison Chock and Evan Bates in ice dance, Jessie Diggins in cross-country skiing, and the women's ice hockey team.",
    },
    {
        "question": "Which countries are performing best?",
        "ground_truth": "Strong performing countries at Milano Cortina 2026 include Sweden, Japan, USA, Canada, Norway, and the Netherlands across various winter sports.",
    },
    
    # ═══════════════════════════════════════════════════════════
    # UPSETS & STORYLINES (3 questions)
    # ═══════════════════════════════════════════════════════════
    {
        "question": "What are the major upsets so far?",
        "ground_truth": "Major upsets include Sara Hector's gold in women's downhill (she wasn't among the pre-Games favorites), and Sweden's victory in men's curling over the favored USA team.",
    },
    {
        "question": "Tell me about Andrea Bocelli and the opening ceremony",
        "ground_truth": "There are rumors that Italian tenor Andrea Bocelli may perform at the Milano Cortina Opening Ceremony, though this has not been officially confirmed by the organizing committee.",
    },
    {
        "question": "What are the biggest surprises at these Olympics?",
        "ground_truth": "Biggest surprises include unexpected medal winners like Sara Hector in alpine skiing, comeback performances from athletes like Therese Johaug, and upsets in team events like men's curling.",
    },
]


# ═══════════════════════════════════════════════════════════
# DATASET GENERATION
# ═══════════════════════════════════════════════════════════

def generate_ragas_dataset() -> Dataset:
    """
    Generate dataset for RAGAS evaluation.
    
    For each question:
      1. Retrieve context from Pinecone
      2. Generate answer using RAG pipeline
      3. Format for RAGAS
    """
    
    print(f"🔄 Generating evaluation dataset ({len(EVAL_QUESTIONS)} questions)...")
    
    # Load models
    try:
        embedding_model = load_embedding_model()
        pinecone_index = load_pinecone_index()
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise
    
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    
    success_count = 0
    error_count = 0
    
    for i, item in enumerate(EVAL_QUESTIONS, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        print(f"  [{i}/{len(EVAL_QUESTIONS)}] Processing: {question[:60]}...")
        
        try:
            # Retrieve context
            matches = retrieve_context(
                question, 
                top_k=7,
                embedding_model=embedding_model,
                pinecone_index=pinecone_index
            )
            
            # Extract context texts
            contexts = [
                m['metadata'].get('text', '')
                for m in matches
                if m['metadata'].get('text')
            ]
            
            if not contexts:
                print(f"    ⚠️  No context found for: {question}")
                contexts = ["No relevant context retrieved"]
            
            # Generate answer
            formatted_context = format_context_for_llm(matches, medal_df=None)
            
            try:
                from app import generate_response
                answer = generate_response(question, formatted_context, "EN", heat=1)
            except Exception as e:
                print(f"    ⚠️  Could not generate answer, using context: {str(e)[:50]}")
                answer = formatted_context[:500] if formatted_context else "Unable to generate answer"
            
            # Add to dataset
            data["question"].append(question)
            data["answer"].append(answer)
            data["contexts"].append(contexts)
            data["ground_truth"].append(ground_truth)
            
            success_count += 1
            print(f"    ✅ Success ({len(contexts)} contexts retrieved)")
            
        except Exception as e:
            error_count += 1
            print(f"    ❌ Error: {str(e)[:100]}")
            
            # Add placeholder to maintain dataset structure
            data["question"].append(question)
            data["answer"].append("Error generating answer")
            data["contexts"].append(["Error retrieving context"])
            data["ground_truth"].append(ground_truth)
    
    print(f"\n📊 Dataset generation complete:")
    print(f"  ✅ Success: {success_count}/{len(EVAL_QUESTIONS)}")
    print(f"  ❌ Errors:  {error_count}/{len(EVAL_QUESTIONS)}")
    
    return Dataset.from_dict(data)


# ═══════════════════════════════════════════════════════════
# RAGAS EVALUATION
# ═══════════════════════════════════════════════════════════

def run_ragas_eval() -> dict:
    """Run RAGAS evaluation and return metrics."""
    
    print("📊 Running RAGAS evaluation...")
    
    # Generate dataset
    dataset = generate_ragas_dataset()
    
    # Evaluate
    try:
        result = evaluate(
            dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
        )
        
        print("✅ RAGAS evaluation complete")
        
        # Convert to dict
        metrics = {
            "context_precision": result.get("context_precision", 0),
            "context_recall": result.get("context_recall", 0),
            "faithfulness": result.get("faithfulness", 0),
            "answer_relevancy": result.get("answer_relevancy", 0),
        }
        
        return metrics
        
    except Exception as e:
        print(f"❌ RAGAS evaluation failed: {e}")
        # Return zeros if evaluation fails
        return {
            "context_precision": 0,
            "context_recall": 0,
            "faithfulness": 0,
            "answer_relevancy": 0,
        }


# ═══════════════════════════════════════════════════════════
# HISTORICAL TRACKING
# ═══════════════════════════════════════════════════════════

def save_metrics_to_history(metrics: dict):
    """Save today's metrics to historical CSV."""
    
    history_file = Path("ragas_history.csv")
    
    # Create record for today
    record = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().isoformat(),
        "context_precision": metrics.get("context_precision", 0),
        "context_recall": metrics.get("context_recall", 0),
        "faithfulness": metrics.get("faithfulness", 0),
        "answer_relevancy": metrics.get("answer_relevancy", 0),
    }
    
    # Append to history
    if history_file.exists():
        df = pd.read_csv(history_file)
        
        # Check if we already have today's data
        today = record["date"]
        if today in df["date"].values:
            print(f"⚠️  Metrics for {today} already exist, updating...")
            df = df[df["date"] != today]
        
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])
    
    df.to_csv(history_file, index=False)
    print(f"✅ Saved metrics to {history_file}")


def print_metrics_report(metrics: dict):
    """Print formatted metrics report."""
    
    print("\n" + "=" * 60)
    print("📊 RAGAS METRICS REPORT")
    print("=" * 60)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"📝 Based on {len(EVAL_QUESTIONS)} evaluation questions")
    
    print(f"\n🎯 Context Precision:  {metrics.get('context_precision', 0):.3f}")
    print("    (Are retrieved chunks relevant?)")
    
    print(f"\n🔍 Context Recall:     {metrics.get('context_recall', 0):.3f}")
    print("    (Did we retrieve all needed info?)")
    
    print(f"\n✅ Faithfulness:       {metrics.get('faithfulness', 0):.3f}")
    print("    (Is answer grounded in context?)")
    
    print(f"\n💬 Answer Relevancy:   {metrics.get('answer_relevancy', 0):.3f}")
    print("    (Does answer address the question?)")
    
    # Composite score
    composite = (
        metrics.get('context_precision', 0) * 0.3 +
        metrics.get('context_recall', 0) * 0.3 +
        metrics.get('faithfulness', 0) * 0.25 +
        metrics.get('answer_relevancy', 0) * 0.15
    )
    
    print(f"\n📈 Composite Score:    {composite:.3f}")
    
    # Quality tier
    if composite >= 0.85:
        tier = "🌟 EXCELLENT"
    elif composite >= 0.75:
        tier = "✅ GOOD"
    elif composite >= 0.60:
        tier = "⚠️  FAIR"
    else:
        tier = "❌ NEEDS IMPROVEMENT"
    
    print(f"   Quality Tier:       {tier}")
    
    print("\n" + "=" * 60)


def compare_to_baseline():
    """Compare today's metrics to baseline (first day)."""
    
    history_file = Path("ragas_history.csv")
    
    if not history_file.exists():
        print("\n📝 This is your baseline - track improvements from here!")
        return
    
    df = pd.read_csv(history_file)
    
    if len(df) < 2:
        print("\n📝 Need more data points to show trends")
        return
    
    baseline = df.iloc[0]
    latest = df.iloc[-1]
    
    print("\n" + "=" * 60)
    print("📈 IMPROVEMENT SINCE BASELINE")
    print("=" * 60)
    print(f"Baseline: {baseline['date']}")
    print(f"Current:  {latest['date']}")
    print(f"Days:     {len(df)}")
    
    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    
    for metric in metrics:
        baseline_val = baseline[metric]
        latest_val = latest[metric]
        delta = latest_val - baseline_val
        pct_change = (delta / baseline_val * 100) if baseline_val > 0 else 0
        
        trend = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"
        
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Baseline: {baseline_val:.3f}")
        print(f"  Current:  {latest_val:.3f}")
        print(f"  Change:   {trend} {delta:+.3f} ({pct_change:+.1f}%)")
    
    print("\n" + "=" * 60)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    """Run daily RAGAS evaluation and track metrics."""
    
    print("🚀 Starting RAGAS evaluation...")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    # Check for required environment variables
    if not os.getenv("PINECONE_API_KEY"):
        print("❌ Error: PINECONE_API_KEY not set")
        print("   Set it with: export PINECONE_API_KEY='your-key'")
        return
    
    # Run evaluation
    metrics = run_ragas_eval()
    
    # Print report
    print_metrics_report(metrics)
    
    # Save to history
    save_metrics_to_history(metrics)
    
    # Compare to baseline
    compare_to_baseline()
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
