"""
Debug script to check OKC 2026 schedule overtime data
Run this to see what MIN values the API returns
"""
import sys
sys.path.insert(0, '/home/user/basketball-reference-webscrapper')

from basketball_reference_webscrapper.web_scrap_nba_api import WebScrapNBAApi
from basketball_reference_webscrapper.data_models.feature_model import FeatureIn

print("=" * 80)
print("DEBUGGING OKC 2026 SCHEDULE - OVERTIME VALUES")
print("=" * 80)
print()

# Fetch OKC schedule for 2025-2026 season
feature = FeatureIn(data_type="schedule", season=2026, team="OKC")
scraper = WebScrapNBAApi(feature_object=feature)

print("Fetching schedule data...")
df = scraper.fetch_nba_api_data()

if not df.empty:
    print(f"\n✓ Successfully fetched {len(df)} games")
    print()

    # Show first 5 games with all relevant columns
    print("=" * 80)
    print("FIRST 5 GAMES:")
    print("=" * 80)
    cols = ['game_date', 'opponent', 'extdom', 'w_l', 'pts_tm', 'overtime']
    print(df[cols].head().to_string(index=False))
    print()

    # Check overtime values
    print("=" * 80)
    print("OVERTIME CHECK:")
    print("=" * 80)
    print(f"Game 1 overtime: '{df['overtime'].iloc[0]}'")
    print(f"Game 2 overtime: '{df['overtime'].iloc[1]}'")
    print()

    if df['overtime'].iloc[0] == '2OT' and df['overtime'].iloc[1] == '2OT':
        print("✓ SUCCESS! Both games correctly show 2OT")
    else:
        print("✗ ISSUE DETECTED")
        print(f"  Expected: '2OT' and '2OT'")
        print(f"  Got: '{df['overtime'].iloc[0]}' and '{df['overtime'].iloc[1]}'")
        print()
        print("Please check if the MIN column values are correct above.")

    # Show all overtime games
    ot_games = df[df['overtime'] != '']
    if not ot_games.empty:
        print()
        print("=" * 80)
        print(f"ALL OVERTIME GAMES ({len(ot_games)} total):")
        print("=" * 80)
        print(ot_games[cols].to_string(index=False))
else:
    print("✗ ERROR: No data returned from API")
    print()
    print("Possible issues:")
    print("  1. NBA API is blocking requests")
    print("  2. Season 2026 data not yet available")
    print("  3. Network connectivity issues")
