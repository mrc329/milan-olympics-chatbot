"""
Milano Cortina 2026 - Daily Nationality Validation (Simplified)
================================================================
Purpose: Works with your EXISTING Pinecone index
No need to recreate the database - just validates and updates metadata
Run via GitHub Actions daily during the Games (Feb 6-22, 2026)
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List
import wikipediaapi
from pinecone import Pinecone

# ============================================================================
# CONFIGURATION
# ============================================================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "milan-2026-olympics"  # Your existing index

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='MilanoCortina2026Bot/1.0',
    language='en'
)

VALID_OLYMPIC_COUNTRIES = {
    'USA': 'United States', 'RUS': 'Russia', 'CAN': 'Canada', 'ITA': 'Italy',
    'GER': 'Germany', 'FRA': 'France', 'JPN': 'Japan', 'CHN': 'China',
    'KOR': 'South Korea', 'GBR': 'Great Britain', 'NOR': 'Norway',
    'SWE': 'Sweden', 'FIN': 'Finland', 'SUI': 'Switzerland', 'AUT': 'Austria',
    'NED': 'Netherlands', 'BEL': 'Belgium', 'CZE': 'Czech Republic',
    'POL': 'Poland', 'AUS': 'Australia', 'ESP': 'Spain', 'UKR': 'Ukraine'
}

# ============================================================================
# FETCH ATHLETES FROM YOUR EXISTING INDEX
# ============================================================================

def fetch_all_athletes_from_pinecone() -> List[Dict]:
    """
    Fetch all athlete vectors from your existing Pinecone index
    Returns list of {id, metadata}
    """
    print(f"Fetching athletes from existing index: {PINECONE_INDEX_NAME}...")
    
    try:
        # Get index stats first
        stats = index.describe_index_stats()
        print(f"Index stats: {stats}")
        
        namespace = 'athletes'  # Change if your namespace is different
        
        # Method 1: Query with dummy vector to get all results
        # This works for indexes with < 10,000 vectors
        dummy_vector = [0.0] * 1536  # Adjust if your dimension is different
        
        results = index.query(
            vector=dummy_vector,
            top_k=10000,
            include_metadata=True,
            namespace=namespace
        )
        
        athletes = []
        for match in results['matches']:
            # Only include if it has athlete metadata
            if match.get('metadata') and match['metadata'].get('name'):
                athletes.append({
                    'id': match['id'],
                    'metadata': match['metadata'],
                    'score': match.get('score', 0)
                })
        
        print(f"✓ Found {len(athletes)} athletes in index\n")
        
        if len(athletes) == 0:
            print("⚠️  No athletes found. Check:")
            print(f"   - Index name: {PINECONE_INDEX_NAME}")
            print(f"   - Namespace: {namespace}")
            print(f"   - Metadata structure (needs 'name' field)")
        
        return athletes
    
    except Exception as e:
        print(f"❌ Error fetching from Pinecone: {e}")
        return []

# ============================================================================
# WIKIPEDIA VALIDATION (Same as before)
# ============================================================================

def check_nationality_on_wikipedia(athlete_name: str, 
                                   expected_country: str) -> Dict:
    """Check athlete's current nationality on Wikipedia"""
    try:
        page = wiki_wiki.page(athlete_name)
        
        if not page.exists():
            page = wiki_wiki.page(f"{athlete_name} figure skater")
        
        if not page.exists():
            return {
                'current_country': expected_country,
                'is_changed': False,
                'confidence': 'page_not_found',
                'message': f"Wikipedia page not found - keeping {expected_country}"
            }
        
        summary = page.summary.lower()
        expected_country_name = VALID_OLYMPIC_COUNTRIES.get(expected_country, '').lower()
        
        if expected_country_name in summary[:500]:
            return {
                'current_country': expected_country,
                'is_changed': False,
                'confidence': 'high',
                'message': f"✓ Confirmed: {athlete_name} still represents {expected_country}"
            }
        
        # Check for OTHER countries
        found_countries = []
        for code, full_name in VALID_OLYMPIC_COUNTRIES.items():
            if code != expected_country and full_name.lower() in summary[:500]:
                found_countries.append(code)
        
        if found_countries:
            new_country = found_countries[0]
            return {
                'current_country': new_country,
                'is_changed': True,
                'confidence': 'medium',
                'message': f"⚠️  CHANGE DETECTED: {athlete_name} may now represent {new_country} (was {expected_country})"
            }
        else:
            return {
                'current_country': expected_country,
                'is_changed': False,
                'confidence': 'low',
                'message': f"⚠️  Unable to confirm {athlete_name}'s nationality - keeping {expected_country}"
            }
    
    except Exception as e:
        return {
            'current_country': expected_country,
            'is_changed': False,
            'confidence': 'error',
            'message': f"Error checking {athlete_name}: {str(e)}"
        }

# ============================================================================
# UPDATE METADATA IN PINECONE
# ============================================================================

def update_athlete_in_pinecone(athlete_id: str, 
                               new_country: str,
                               old_metadata: Dict) -> bool:
    """
    Update athlete's nationality metadata in Pinecone
    Re-upserts the same vector with updated metadata
    """
    try:
        # Fetch the original vector
        fetch_response = index.fetch(ids=[athlete_id])
        
        if athlete_id not in fetch_response['vectors']:
            print(f"  ❌ Vector not found: {athlete_id}")
            return False
        
        original_vector = fetch_response['vectors'][athlete_id]['values']
        
        # Update metadata
        updated_metadata = old_metadata.copy()
        updated_metadata['country'] = new_country
        updated_metadata['country_full'] = VALID_OLYMPIC_COUNTRIES[new_country]
        updated_metadata['is_home_athlete'] = (new_country == 'ITA')
        updated_metadata['last_validated'] = datetime.utcnow().isoformat()
        updated_metadata['nationality_updated'] = True
        updated_metadata['previous_country'] = old_metadata.get('country')
        
        # Re-upsert with same vector, updated metadata
        index.upsert(
            vectors=[{
                'id': athlete_id,
                'values': original_vector,
                'metadata': updated_metadata
            }]
        )
        
        print(f"  ✓ Updated {athlete_id}: {old_metadata.get('country')} → {new_country}")
        return True
    
    except Exception as e:
        print(f"  ❌ Failed to update {athlete_id}: {str(e)}")
        return False

# ============================================================================
# MAIN VALIDATION PIPELINE
# ============================================================================

def run_daily_validation():
    """Main pipeline: Check all athletes, update if nationality changed"""
    
    print("\n" + "="*80)
    print(f"DAILY NATIONALITY VALIDATION - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Index: {PINECONE_INDEX_NAME}")
    print("="*80 + "\n")
    
    # Step 1: Fetch all athletes from YOUR existing index
    athletes = fetch_all_athletes_from_pinecone()
    
    if not athletes:
        print("❌ No athletes found in index. Exiting.")
        return 0
    
    # Step 2: Validate each athlete
    changes_detected = []
    validation_results = []
    
    print("Validating nationalities via Wikipedia...\n")
    print("-"*80)
    
    for athlete in athletes:
        athlete_name = athlete['metadata'].get('name', 'Unknown')
        current_country = athlete['metadata'].get('country', 'UNK')
        
        print(f"Checking: {athlete_name} ({current_country})...")
        
        # Check Wikipedia
        result = check_nationality_on_wikipedia(athlete_name, current_country)
        print(f"  {result['message']}")
        
        validation_results.append({
            'athlete': athlete_name,
            'athlete_id': athlete['id'],
            'expected_country': current_country,
            'found_country': result['current_country'],
            'is_changed': result['is_changed'],
            'confidence': result['confidence'],
            'timestamp': datetime.utcnow().isoformat()
        })
        
        if result['is_changed']:
            change_data = {
                'athlete_id': athlete['id'],
                'athlete_name': athlete_name,
                'old_country': current_country,
                'new_country': result['current_country'],
                'confidence': result['confidence'],
                'metadata': athlete['metadata']
            }
            
            # Only auto-update if confidence is medium or high
            if result['confidence'] in ['high', 'medium']:
                changes_detected.append(change_data)
                print(f"  → Will auto-update (confidence: {result['confidence']})")
            else:
                # Low confidence - flag for manual review
                print(f"  → SKIPPING auto-update (confidence: {result['confidence']})")
                print(f"  → Requires manual review via GitHub issue")
                validation_results[-1]['needs_manual_review'] = True
                validation_results[-1]['reason'] = 'Low confidence nationality change'
        
        time.sleep(0.5)  # Rate limit
    
    print("-"*80)
    
    # Step 3: Apply updates if needed
    if changes_detected:
        print(f"\n⚠️  NATIONALITY CHANGES DETECTED: {len(changes_detected)} athletes")
        print("="*80)
        print("Updating Pinecone database...\n")
        
        for change in changes_detected:
            print(f"Updating: {change['athlete_name']} ({change['old_country']} → {change['new_country']})")
            update_athlete_in_pinecone(
                athlete_id=change['athlete_id'],
                new_country=change['new_country'],
                old_metadata=change['metadata']
            )
        
        print("\n" + "="*80)
    else:
        print(f"\n✓ NO CHANGES DETECTED - All nationalities confirmed")
    
    # Step 4: Save log
    log_data = {
        'validation_timestamp': datetime.utcnow().isoformat(),
        'index_name': PINECONE_INDEX_NAME,
        'total_athletes_checked': len(athletes),
        'changes_detected': len(changes_detected),
        'validation_results': validation_results,
        'nationality_changes': changes_detected
    }
    
    log_filename = f"validation_log_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_filename, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n✓ Validation log saved to: {log_filename}")
    
    # Step 5: Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Index: {PINECONE_INDEX_NAME}")
    print(f"Athletes checked: {len(athletes)}")
    print(f"Changes detected: {len(changes_detected)}")
    print(f"Database updated: {'Yes' if changes_detected else 'No'}")
    print(f"Log file: {log_filename}")
    print(f"{'='*80}\n")
    
    # Return exit code (1 = changes detected for GitHub Actions)
    return 1 if changes_detected else 0

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    exit_code = run_daily_validation()
    exit(exit_code)
