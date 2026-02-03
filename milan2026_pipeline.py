"""
fix_pipeline_indentation.py
────────────────────────────
Script to identify and suggest fixes for indentation errors in milan2026_pipeline.py

Run this in your repo:
    python fix_pipeline_indentation.py
"""

import re

COMMON_INDENTATION_FIXES = """
COMMON INDENTATION ERROR PATTERN IN YOUR FILE:
───────────────────────────────────────────────

The error is on line 1479 after an 'if' statement on line 1478.
This is typically caused by missing indentation after an if/for/while/try block.

PATTERN TO FIND (around line 1478-1479):
    if some_condition:
    col_map = {}  # ← WRONG: needs 4 more spaces

SHOULD BE:
    if some_condition:
        col_map = {}  # ← CORRECT: indented 4 spaces


LIKELY LOCATION IN milan2026_pipeline.py:
─────────────────────────────────────────

Based on the app.py fix we did, this is probably in a section that processes
medal data or dataframes. Look for code similar to:

    if medal_df is not None and not medal_df.empty:
    col_map = {}  # ← MISSING INDENTATION
    for c in medal_df.columns:
        ...

Or possibly in a section like:

    if during_games:
        st.markdown(f'<div class="sidebar-heading">...</div>')
    
        if medal_df is not None and not medal_df.empty:
        col_map = {}  # ← MISSING INDENTATION


QUICK FIX STEPS:
────────────────

1. Open milan2026_pipeline.py in your editor
2. Go to line 1478-1479
3. Check if line 1479 and following lines need 4 more spaces of indentation
4. Add the indentation and save
5. Test with: python -m py_compile milan2026_pipeline.py


AUTOMATED FIX PATTERN:
──────────────────────

If you see this pattern:

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

Change it to:

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

Notice: Every line after the 'if' gets 4 additional spaces.
"""

def check_file_exists():
    """Check if milan2026_pipeline.py exists."""
    import os
    if os.path.exists("milan2026_pipeline.py"):
        return True
    print("❌ milan2026_pipeline.py not found in current directory")
    print("   Run this script from your repository root")
    return False

def show_error_context():
    """Show the exact lines around the error."""
    try:
        with open("milan2026_pipeline.py", "r") as f:
            lines = f.readlines()
        
        print("\n" + "="*60)
        print("LINES 1475-1485 (around the error):")
        print("="*60)
        
        for i in range(1474, min(1485, len(lines))):
            line_num = i + 1
            line = lines[i].rstrip('\n')
            marker = " ← ERROR HERE" if line_num == 1479 else ""
            print(f"{line_num:4d} | {line}{marker}")
        
        print("\n" + "="*60)
        print("ANALYSIS:")
        print("="*60)
        
        if len(lines) > 1478:
            line_1478 = lines[1477].strip()
            line_1479 = lines[1478]
            
            print(f"Line 1478: {line_1478}")
            print(f"Line 1479: {repr(line_1479)}")
            
            spaces_1478 = len(lines[1477]) - len(lines[1477].lstrip())
            spaces_1479 = len(lines[1478]) - len(lines[1478].lstrip())
            
            print(f"\nIndentation on line 1478: {spaces_1478} spaces")
            print(f"Indentation on line 1479: {spaces_1479} spaces")
            
            if line_1478.endswith(':'):
                expected_indent = spaces_1478 + 4
                print(f"\n⚠️  Line 1478 ends with ':', so line 1479 should have {expected_indent} spaces")
                print(f"   But it has {spaces_1479} spaces")
                print(f"   → Add {expected_indent - spaces_1479} more spaces to line 1479 and following lines")
        
        return True
    except FileNotFoundError:
        print("❌ Could not read milan2026_pipeline.py")
        return False
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        return False

def main():
    print(COMMON_INDENTATION_FIXES)
    
    if not check_file_exists():
        return
    
    print("\n" + "="*60)
    print("CHECKING YOUR FILE...")
    print("="*60)
    
    if show_error_context():
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Edit milan2026_pipeline.py")
        print("2. Add the suggested indentation to lines 1479+")
        print("3. Save the file")
        print("4. Test: python -m py_compile milan2026_pipeline.py")
        print("5. If that works: python milan2026_pipeline.py (to test it runs)")

if __name__ == "__main__":
    main()
