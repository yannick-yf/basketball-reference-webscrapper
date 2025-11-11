"""
Example usage script for the WebScrapNBAApi class.

This script demonstrates how to use the NBA API scraper to fetch
gamelog and schedule data for NBA teams.
"""

from basketball_reference_webscrapper.data_models.feature_model import FeatureIn
from basketball_reference_webscrapper.web_scrap_nba_api import WebScrapNBAApi
import pandas as pd


def example_single_team_gamelog():
    """Example: Fetch gamelog data for a single team."""
    print("\n" + "="*80)
    print("Example 1: Fetching gamelog data for Boston Celtics (2024 season)")
    print("="*80)

    feature = FeatureIn(
        data_type="gamelog",
        season=2024,
        team="BOS"
    )

    scraper = WebScrapNBAApi(feature_object=feature)
    df = scraper.fetch_nba_api_data()

    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")

    return df


def example_multiple_teams_gamelog():
    """Example: Fetch gamelog data for multiple teams."""
    print("\n" + "="*80)
    print("Example 2: Fetching gamelog data for Lakers and Warriors (2023 season)")
    print("="*80)

    feature = FeatureIn(
        data_type="gamelog",
        season=2023,
        team=["LAL", "GSW"]
    )

    scraper = WebScrapNBAApi(feature_object=feature)
    df = scraper.fetch_nba_api_data()

    print(f"\nShape: {df.shape}")
    print(f"\nTeams in data: {df['tm'].unique()}")
    print(f"\nSample data:\n{df.head(10)}")

    return df


def example_single_team_schedule():
    """Example: Fetch schedule data for a single team."""
    print("\n" + "="*80)
    print("Example 3: Fetching schedule data for Miami Heat (2024 season)")
    print("="*80)

    feature = FeatureIn(
        data_type="schedule",
        season=2024,
        team="MIA"
    )

    scraper = WebScrapNBAApi(feature_object=feature)
    df = scraper.fetch_nba_api_data()

    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")

    return df


def example_all_teams_schedule():
    """Example: Fetch schedule data for all teams (use with caution - many API calls)."""
    print("\n" + "="*80)
    print("Example 4: Fetching schedule data for all teams (2024 season)")
    print("WARNING: This will make 30+ API calls and take several minutes!")
    print("="*80)

    # Uncomment below to run
    # feature = FeatureIn(
    #     data_type="schedule",
    #     season=2024,
    #     team="all"
    # )
    #
    # scraper = WebScrapNBAApi(feature_object=feature)
    # df = scraper.fetch_nba_api_data()
    #
    # print(f"\nShape: {df.shape}")
    # print(f"\nNumber of teams: {df['tm'].nunique()}")
    # print(f"\nTeams: {sorted(df['tm'].unique())}")
    #
    # return df

    print("\n(Example commented out to avoid long execution time)")
    return None


def example_comparison_with_basketball_reference():
    """Example: Compare NBA API data with Basketball Reference data."""
    print("\n" + "="*80)
    print("Example 5: Comparing NBA API vs Basketball Reference data")
    print("="*80)

    from basketball_reference_webscrapper.webscrapping_basketball_reference import (
        WebScrapBasketballReference
    )

    # Fetch from NBA API
    feature_nba = FeatureIn(
        data_type="gamelog",
        season=2024,
        team="BOS"
    )
    scraper_nba = WebScrapNBAApi(feature_object=feature_nba)
    df_nba = scraper_nba.fetch_nba_api_data()

    print("\nNBA API Data:")
    print(f"Shape: {df_nba.shape}")
    print(f"Columns: {list(df_nba.columns)}")

    # Fetch from Basketball Reference (commented out to avoid actual scraping)
    # feature_br = FeatureIn(
    #     data_type="gamelog",
    #     season=2024,
    #     team="BOS"
    # )
    # scraper_br = WebScrapBasketballReference(feature_object=feature_br)
    # df_br = scraper_br.webscrappe_nba_games_data()
    #
    # print("\nBasketball Reference Data:")
    # print(f"Shape: {df_br.shape}")
    # print(f"Columns: {list(df_br.columns)}")
    #
    # # Compare column names
    # common_cols = set(df_nba.columns).intersection(set(df_br.columns))
    # print(f"\nCommon columns: {len(common_cols)}")
    # print(f"NBA API only: {set(df_nba.columns) - set(df_br.columns)}")
    # print(f"Basketball Reference only: {set(df_br.columns) - set(df_nba.columns)}")

    print("\n(Basketball Reference comparison commented out)")

    return df_nba


def example_error_handling():
    """Example: Demonstrate error handling."""
    print("\n" + "="*80)
    print("Example 6: Error handling examples")
    print("="*80)

    # Invalid data type
    try:
        print("\nTesting invalid data_type...")
        feature = FeatureIn(
            data_type="invalid_type",
            season=2024,
            team="BOS"
        )
        scraper = WebScrapNBAApi(feature_object=feature)
        df = scraper.fetch_nba_api_data()
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")

    # Invalid season
    try:
        print("\nTesting invalid season...")
        feature = FeatureIn(
            data_type="gamelog",
            season=1990,
            team="BOS"
        )
        scraper = WebScrapNBAApi(feature_object=feature)
        df = scraper.fetch_nba_api_data()
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")

    # Invalid team
    try:
        print("\nTesting invalid team abbreviation...")
        feature = FeatureIn(
            data_type="gamelog",
            season=2024,
            team="INVALID"
        )
        scraper = WebScrapNBAApi(feature_object=feature)
        df = scraper.fetch_nba_api_data()
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")

    print("\n✓ All error handling tests passed!")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("NBA API Scraper - Example Usage")
    print("="*80)

    # Run examples
    example_single_team_gamelog()
    # example_multiple_teams_gamelog()  # Commented to reduce API calls
    # example_single_team_schedule()  # Commented to reduce API calls
    # example_all_teams_schedule()  # Commented - very long execution
    # example_comparison_with_basketball_reference()  # Commented to avoid web scraping
    example_error_handling()

    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)
    print("\nNotes:")
    print("- The NBA API has rate limits (~100 requests/minute)")
    print("- Some examples are commented out to avoid excessive API calls")
    print("- Uncomment examples in the code to test specific functionality")
    print("- For production use, consider implementing caching mechanisms")
    print("="*80)


if __name__ == "__main__":
    main()
