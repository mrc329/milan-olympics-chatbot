"""
Milano Cortina 2026 - Part 2: Daily Nationality Validation
===========================================================
Purpose: Runs daily during Games (Feb 6-22, 2026) via GitLab CI
Checks all athletes in Pinecone against Wikipedia for nationality changes
Updates Pinecone if discrepancies found
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
import wikipediaapi
from pinecone import Pinecone

# ============================================================================
# CONFIGURATION
# ============================================================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "milano-cortina-2026"

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
# FETCH ALL ATHLETES FROM PINECONE
# ============================================================================

def fetch_all_athletes_from_pinecone() -> List[Dict]:
    """
    Fetch all athlete vectors from Pinecone
    Returns list of {id, metadata}
    """
    print("Fetching all athletes from Pinecone...")
    
    # Query with empty vector to get all results
    # (Pinecone doesn't have a "list all" API, so we use a dummy query)
    results = index.query(
        vector=[0.0] * 1536,  # Dummy vector
        top_k=10000,  # Max results
        include_metadata=True,
        namespace='athletes'
    )
    
    athletes = []
    for match in results['matches']:
        athletes.append({
            'id': match['id'],
            'metadata': match['metadata']
        })
    
    print(f"✓ Found {len(athletes)} athletes in database\\n")
    return athletes

# ============================================================================
# VALIDATE NATIONALITY VIA WIKIPEDIA
# ============================================================================

def check_nationality_on_wikipedia(athlete_name: str, 
                                   expected_country: str) -> Dict:
    """
    Check athlete's current nationality on Wikipedia
    Returns: {'current_country': str, 'is_changed': bool, 'confidence': str}
    """
    try:
        # Try main page first
        page = wiki_wiki.page(athlete_name)
        
        if not page.exists():
            # Try with sport suffix
            page = wiki_wiki.page(f"{athlete_name} figure skater")
        
        if not page.exists():
            return {
                'current_country': expected_country,
                'is_changed': False,
                'confidence': 'page_not_found',
                'message': f"Wikipedia page not found - keeping {expected_country}"
            }
        
        # Check for expected country in summary
        summary = page.summary.lower()
        expected_country_name = VALID_OLYMPIC_COUNTRIES.get(expected_country, '').lower()
        
        if expected_country_name in summary[:500]:
            return {
                'current_country': expected_country,
                'is_changed': False,
                'confidence': 'high',
                'message': f"✓ Confirmed: {athlete_name} still represents {expected_country}"
            }
        
        # Country not found - check for OTHER countries
        found_countries = []
        for code, full_name in VALID_OLYMPIC_COUNTRIES.items():
            if code != expected_country and full_name.lower() in summary[:500]:
                found_countries.append(code)
        
        if found_countries:
            # Possible nationality change detected!
            new_country = found_countries[0]
            return {
                'current_country': new_country,
                'is_changed': True,
                'confidence': 'medium',
                'message': f"⚠️  CHANGE DETECTED: {athlete_name} may now represent {new_country} (was {expected_country})"
            }
        else:
            # Unclear - keep existing
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
# UPDATE PINECONE METADATA
# ============================================================================

def update_athlete_nationality_in_pinecone(athlete_id: str, 
                                          new_country: str,
                                          old_metadata: Dict) -> bool:
    """
    Update athlete's nationality in Pinecone metadata
    Preserves all other metadata fields
    """
    try:
        # Update metadata
        updated_metadata = old_metadata.copy()
        updated_metadata['country'] = new_country
        updated_metadata['country_full'] = VALID_OLYMPIC_COUNTRIES[new_country]
        updated_metadata['is_home_athlete'] = (new_country == 'ITA')
        updated_metadata['last_validated'] = datetime.utcnow().isoformat()
        updated_metadata['nationality_updated'] = True
        updated_metadata['previous_country'] = old_metadata['country']
        
        # Update in Pinecone
        # Note: We need to re-upsert with the same vector but updated metadata
        # Fetch the original vector first
        fetch_response = index.fetch(ids=[athlete_id], namespace='athletes')
        
        if athlete_id not in fetch_response['vectors']:
            print(f"  ❌ Vector not found: {athlete_id}")
            return False
        
        original_vector = fetch_response['vectors'][athlete_id]['values']
        
        # Upsert with updated metadata
        index.upsert(
            vectors=[{
                'id': athlete_id,
                'values': original_vector,
                'metadata': updated_metadata
            }],
            namespace='athletes'
        )
        
        print(f"  ✓ Updated {athlete_id}: {old_metadata['country']} → {new_country}")
        return True
    
    except Exception as e:
        print(f"  ❌ Failed to update {athlete_id}: {str(e)}")
        return False

# ============================================================================
# MAIN VALIDATION PIPELINE
# ============================================================================

def run_daily_validation():
    """
    Main pipeline: Check all athletes, update if nationality changed
    """
    print("\\n" + "="*80)
    print(f"DAILY NATIONALITY VALIDATION - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("="*80 + "\\n")
    
    # Step 1: Fetch all athletes from Pinecone
    athletes = fetch_all_athletes_from_pinecone()
    
    if not athletes:
        print("❌ No athletes found in database. Run initial ingestion first.")
        return
    
    # Step 2: Check each athlete's nationality
    changes_detected = []
    validation_results = []
    
    print("Validating nationalities via Wikipedia...\\n")
    print("-"*80)
    
    for athlete in athletes:
        athlete_name = athlete['metadata']['name']
        current_country = athlete['metadata']['country']
        
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
                validation_results.append({
                    **validation_results[-1],
                    'needs_manual_review': True,
                    'reason': 'Low confidence nationality change'
                })
        
        # Rate limit: sleep between requests
        time.sleep(0.5)
    
    print("-"*80)
    
    # Step 3: Update Pinecone if changes detected
    if changes_detected:
        print(f"\\n⚠️  NATIONALITY CHANGES DETECTED: {len(changes_detected)} athletes")
        print("="*80)
        print("Updating Pinecone database...\\n")
        
        update_success = []
        update_failed = []
        
        for change in changes_detected:
            print(f"Updating: {change['athlete_name']} ({change['old_country']} → {change['new_country']})")
            
            success = update_athlete_nationality_in_pinecone(
                athlete_id=change['athlete_id'],
                new_country=change['new_country'],
                old_metadata=change['metadata']
            )
            
            if success:
                update_success.append(change)
            else:
                update_failed.append(change)
        
        print("\\n" + "="*80)
        print(f"✓ Successfully updated: {len(update_success)}")
        print(f"❌ Failed to update: {len(update_failed)}")
        print("="*80)
        
    else:
        print(f"\\n✓ NO CHANGES DETECTED - All nationalities confirmed")
    
    # Step 4: Save validation log
    log_data = {
        'validation_timestamp': datetime.utcnow().isoformat(),
        'total_athletes_checked': len(athletes),
        'changes_detected': len(changes_detected),
        'changes_applied': len([c for c in changes_detected if c]) if changes_detected else 0,
        'validation_results': validation_results,
        'nationality_changes': changes_detected
    }
    
    log_filename = f"validation_log_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_filename, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\\n✓ Validation log saved to: {log_filename}")
    
    # Step 5: Summary
    print(f"\\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Athletes checked: {len(athletes)}")
    print(f"Changes detected: {len(changes_detected)}")
    print(f"Database updated: {'Yes' if changes_detected else 'No'}")
    print(f"Log file: {log_filename}")
    print(f"{'='*80}\\n")
    
    # Return exit code (0 = success, 1 = changes detected for CI alerting)
    return 1 if changes_detected else 0

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    exit_code = run_daily_validation()
    exit(exit_code)
