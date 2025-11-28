"""Test WebScrapNBAApiBoxscore"""

import json
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd

from basketball_reference_webscrapper.data_models.feature_model import FeatureIn
from basketball_reference_webscrapper.web_scrap_player_boxscore_nba_api import (
    FailedGameRecord,
    WebScrapNBAApiBoxscore,
)


class TestWebScrapNBAApiBoxscore(TestCase):
    """Tests for the WebScrapNBAApiBoxscore class that fetches boxscore data from NBA API."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.valid_season = 2024
        self.valid_team = "BOS"
        self.valid_feature = FeatureIn(
            data_type="boxscore",
            season=self.valid_season,
            team=self.valid_team
        )

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        for file in Path(self.temp_dir).glob("*"):
            file.unlink()
        Path(self.temp_dir).rmdir()

    # -------------------------------------------------------------------------
    # Initialization Tests
    # -------------------------------------------------------------------------

    def test_initialization(self):
        """
        GIVEN valid FeatureIn arguments
        WHEN WebScrapNBAApiBoxscore is initialized
        THEN it creates an instance successfully with correct attributes
        """
        scraper = WebScrapNBAApiBoxscore(
            feature_object=self.valid_feature,
            cache_dir=Path(self.temp_dir)
        )

        assert scraper is not None
        assert scraper.feature_object.data_type == "boxscore"
        assert scraper.feature_object.season == self.valid_season
        assert scraper.feature_object.team == self.valid_team
        assert scraper._request_count == 0
        assert len(scraper._fetched_game_ids) == 0

    def test_initialization_creates_cache_dir(self):
        """
        GIVEN a non-existent cache directory path
        WHEN WebScrapNBAApiBoxscore is initialized
        THEN it creates the cache directory
        """
        new_cache_dir = Path(self.temp_dir) / "new_cache"
        assert not new_cache_dir.exists()

        scraper = WebScrapNBAApiBoxscore(
            feature_object=self.valid_feature,
            cache_dir=new_cache_dir
        )

        assert new_cache_dir.exists()
        new_cache_dir.rmdir()

    # -------------------------------------------------------------------------
    # Team Mapping Tests
    # -------------------------------------------------------------------------

    def test_team_mapping_exists(self):
        """
        GIVEN a WebScrapNBAApiBoxscore instance
        WHEN checking team_mapping attribute
        THEN it contains valid NBA team IDs
        """
        scraper = WebScrapNBAApiBoxscore(
            feature_object=self.valid_feature,
            cache_dir=Path(self.temp_dir)
        )

        assert scraper.team_mapping is not None
        assert 'BOS' in scraper.team_mapping
        assert scraper.team_mapping['BOS']['team_id'] == 1610612738

    def test_get_nba_team_id_valid(self):
        """
        GIVEN valid team abbreviations
        WHEN _get_nba_team_id is called
        THEN it returns correct team IDs
        """
        scraper = WebScrapNBAApiBoxscore(
            feature_object=self.valid_feature,
            cache_dir=Path(self.temp_dir)
        )

        bos_id = scraper._get_nba_team_id('BOS')
        lal_id = scraper._get_nba_team_id('LAL')

        assert bos_id == 1610612738
        assert lal_id == 1610612747

    def test_get_nba_team_id_invalid(self):
        """
        GIVEN an invalid team abbreviation
        WHEN _get_nba_team_id is called
        THEN it returns None
        """
        scraper = WebScrapNBAApiBoxscore(
            feature_object=self.valid_feature,
            cache_dir=Path(self.temp_dir)
        )

        invalid_id = scraper._get_nba_team_id('INVALID')

        assert invalid_id is None

    # -------------------------------------------------------------------------
    # Season Format Conversion Tests
    # -------------------------------------------------------------------------

    def test_convert_season_format(self):
        """
        GIVEN a season year
        WHEN _convert_season_format is called
        THEN it returns correct NBA API format
        """
        scraper = WebScrapNBAApiBoxscore(
            feature_object=self.valid_feature,
            cache_dir=Path(self.temp_dir)
        )

        assert scraper._convert_season_format(2024) == "2023-24"
        assert scraper._convert_season_format(2023) == "2022-23"
        assert scraper._convert_season_format(2000) == "1999-00"

    # -------------------------------------------------------------------------
    # Validation Tests
    # -------------------------------------------------------------------------

    def test_data_type_validation_valid(self):
        """
        GIVEN valid data_type
        WHEN _get_data_type_validation is called
        THEN no error is raised
        """
        feature = FeatureIn(data_type="boxscore", season=self.valid_season, team=self.valid_team)
        scraper = WebScrapNBAApiBoxscore(feature_object=feature, cache_dir=Path(self.temp_dir))

        scraper._get_data_type_validation()  # Should not raise

    def test_data_type_validation_invalid(self):
        """
        GIVEN invalid data_type
        WHEN _get_data_type_validation is called
        THEN ValueError is raised
        """
        feature = FeatureIn(data_type="invalid_type", season=self.valid_season, team=self.valid_team)
        scraper = WebScrapNBAApiBoxscore(feature_object=feature, cache_dir=Path(self.temp_dir))

        with self.assertRaises(ValueError) as context:
            scraper._get_data_type_validation()

        assert "not supported" in str(context.exception)

    def test_season_validation_valid(self):
        """
        GIVEN valid season (>= 2000)
        WHEN _get_season_validation is called
        THEN no error is raised
        """
        feature = FeatureIn(data_type="boxscore", season=2023, team=self.valid_team)
        scraper = WebScrapNBAApiBoxscore(feature_object=feature, cache_dir=Path(self.temp_dir))

        scraper._get_season_validation()  # Should not raise

    def test_season_validation_invalid(self):
        """
        GIVEN season before 2000
        WHEN _get_season_validation is called
        THEN ValueError is raised
        """
        feature = FeatureIn(data_type="boxscore", season=1998, team=self.valid_team)
        scraper = WebScrapNBAApiBoxscore(feature_object=feature, cache_dir=Path(self.temp_dir))

        with self.assertRaises(ValueError) as context:
            scraper._get_season_validation()

        assert "not supported" in str(context.exception)

    def test_team_validation_single_valid(self):
        """
        GIVEN valid single team abbreviation
        WHEN _get_team_list_values_validation is called
        THEN no error is raised
        """
        feature = FeatureIn(data_type="boxscore", season=self.valid_season, team="BOS")
        scraper = WebScrapNBAApiBoxscore(feature_object=feature, cache_dir=Path(self.temp_dir))

        team_list = ['BOS', 'LAL', 'GSW']
        scraper._get_team_list_values_validation(team_list)  # Should not raise

    def test_team_validation_invalid(self):
        """
        GIVEN invalid team abbreviation
        WHEN _get_team_list_values_validation is called
        THEN ValueError is raised
        """
        feature = FeatureIn(data_type="boxscore", season=self.valid_season, team="INVALID")
        scraper = WebScrapNBAApiBoxscore(feature_object=feature, cache_dir=Path(self.temp_dir))

        with self.assertRaises(ValueError):
            scraper._get_team_list_values_validation(['BOS', 'LAL', 'GSW'])

    # -------------------------------------------------------------------------
    # Rate Limit Detection Tests
    # -------------------------------------------------------------------------

    def test_is_rate_limit_error_detects_429(self):
        """
        GIVEN an exception with rate limit indicators
        WHEN _is_rate_limit_error is called
        THEN it returns True
        """
        scraper = WebScrapNBAApiBoxscore(
            feature_object=self.valid_feature,
            cache_dir=Path(self.temp_dir)
        )

        error_429 = Exception("HTTP Error 429: Too Many Requests")
        error_timeout = Exception("Connection timed out")
        error_reset = Exception("Connection reset by peer")

        assert scraper._is_rate_limit_error(error_429) is True
        assert scraper._is_rate_limit_error(error_timeout) is True
        assert scraper._is_rate_limit_error(error_reset) is True

    def test_is_rate_limit_error_returns_false_for_other_errors(self):
        """
        GIVEN an exception without rate limit indicators
        WHEN _is_rate_limit_error is called
        THEN it returns False
        """
        scraper = WebScrapNBAApiBoxscore(
            feature_object=self.valid_feature,
            cache_dir=Path(self.temp_dir)
        )

        error_value = ValueError("Invalid game ID format")
        error_key = KeyError("GAME_ID column not found")

        assert scraper._is_rate_limit_error(error_value) is False
        assert scraper._is_rate_limit_error(error_key) is False

    # -------------------------------------------------------------------------
    # Failed Games Cache Tests
    # -------------------------------------------------------------------------

    def test_record_failed_game(self):
        """
        GIVEN a game ID and failure reason
        WHEN _record_failed_game is called
        THEN it adds the game to _failed_games dict
        """
        scraper = WebScrapNBAApiBoxscore(
            feature_object=self.valid_feature,
            cache_dir=Path(self.temp_dir)
        )

        scraper._record_failed_game("0022400123", "Connection timeout")

        assert "0022400123" in scraper._failed_games
        record = scraper._failed_games["0022400123"]
        assert record.game_id == "0022400123"
        assert record.failure_reason == "Connection timeout"

    def test_get_failed_games_summary_empty(self):
        """
        GIVEN no failed games
        WHEN get_failed_games_summary is called
        THEN it returns empty DataFrame with correct columns
        """
        scraper = WebScrapNBAApiBoxscore(
            feature_object=self.valid_feature,
            cache_dir=Path(self.temp_dir)
        )

        summary = scraper.get_failed_games_summary()

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 0
        assert list(summary.columns) == ["game_id", "failure_reason", "timestamp"]

    def test_get_failed_games_summary_with_failures(self):
        """
        GIVEN some failed games
        WHEN get_failed_games_summary is called
        THEN it returns DataFrame with all failed games
        """
        scraper = WebScrapNBAApiBoxscore(
            feature_object=self.valid_feature,
            cache_dir=Path(self.temp_dir)
        )

        scraper._record_failed_game("0022400001", "Error 1")
        scraper._record_failed_game("0022400002", "Error 2")

        summary = scraper.get_failed_games_summary()

        assert len(summary) == 2
        assert set(summary["game_id"]) == {"0022400001", "0022400002"}

    def test_save_and_load_failed_games_cache(self):
        """
        GIVEN failed games in the extractor
        WHEN save and load cache methods are called
        THEN it persists and restores the failed games correctly
        """
        scraper = WebScrapNBAApiBoxscore(
            feature_object=self.valid_feature,
            cache_dir=Path(self.temp_dir)
        )

        scraper._record_failed_game("0022400001", "Test error")
        filepath = scraper.save_failed_games_cache("test_cache.json")

        assert Path(filepath).exists()

        # Create new scraper and load cache
        new_scraper = WebScrapNBAApiBoxscore(
            feature_object=self.valid_feature,
            cache_dir=Path(self.temp_dir)
        )
        count = new_scraper.load_failed_games_cache(filepath)

        assert count == 1
        assert "0022400001" in new_scraper._failed_games

    def test_save_failed_games_cache_empty(self):
        """
        GIVEN no failed games
        WHEN save_failed_games_cache is called
        THEN it returns empty string
        """
        scraper = WebScrapNBAApiBoxscore(
            feature_object=self.valid_feature,
            cache_dir=Path(self.temp_dir)
        )

        filepath = scraper.save_failed_games_cache("test_cache.json")

        assert filepath == ""

    # -------------------------------------------------------------------------
    # Session Management Tests
    # -------------------------------------------------------------------------

    def test_get_request_count_initial(self):
        """
        GIVEN a freshly initialized scraper
        WHEN get_request_count is called
        THEN it returns 0
        """
        scraper = WebScrapNBAApiBoxscore(
            feature_object=self.valid_feature,
            cache_dir=Path(self.temp_dir)
        )

        assert scraper.get_request_count() == 0

    def test_check_and_reset_session_increments_count(self):
        """
        GIVEN a scraper with request count below batch_size
        WHEN _check_and_reset_session is called
        THEN it increments the request count
        """
        scraper = WebScrapNBAApiBoxscore(
            feature_object=self.valid_feature,
            cache_dir=Path(self.temp_dir)
        )
        initial_count = scraper._request_count

        scraper._check_and_reset_session()

        assert scraper._request_count == initial_count + 1

    def test_reset_session_preserves_state(self):
        """
        GIVEN a scraper with fetched game IDs and failed games
        WHEN _reset_session is called
        THEN it preserves the fetched IDs and failed games
        """
        scraper = WebScrapNBAApiBoxscore(
            feature_object=self.valid_feature,
            cache_dir=Path(self.temp_dir)
        )

        scraper._fetched_game_ids.add("0022400001")
        scraper._record_failed_game("0022400002", "Test error")
        scraper._request_count = 100

        scraper._reset_session()

        assert scraper._request_count == 0
        assert "0022400001" in scraper._fetched_game_ids
        assert "0022400002" in scraper._failed_games

    # -------------------------------------------------------------------------
    # Filter Teams Tests
    # -------------------------------------------------------------------------

    def test_filter_teams_single(self):
        """
        GIVEN single team in feature_object
        WHEN _filter_teams is called
        THEN it returns only that team
        """
        feature = FeatureIn(data_type="boxscore", season=self.valid_season, team="BOS")
        scraper = WebScrapNBAApiBoxscore(feature_object=feature, cache_dir=Path(self.temp_dir))

        team_refdata = pd.DataFrame({
            'team_abrev': ['BOS', 'LAL', 'GSW'],
            'city': ['Boston', 'Los Angeles', 'Golden State']
        })

        filtered = scraper._filter_teams(team_refdata)

        assert len(filtered) == 1
        assert filtered['team_abrev'].iloc[0] == 'BOS'

    def test_filter_teams_all(self):
        """
        GIVEN team="all" in feature_object
        WHEN _filter_teams is called
        THEN it returns all teams
        """
        feature = FeatureIn(data_type="boxscore", season=self.valid_season, team="all")
        scraper = WebScrapNBAApiBoxscore(feature_object=feature, cache_dir=Path(self.temp_dir))

        team_refdata = pd.DataFrame({
            'team_abrev': ['BOS', 'LAL', 'GSW', 'MIA'],
            'city': ['Boston', 'Los Angeles', 'Golden State', 'Miami']
        })

        filtered = scraper._filter_teams(team_refdata)

        assert len(filtered) == 4

    # -------------------------------------------------------------------------
    # Column Mapping Tests
    # -------------------------------------------------------------------------

    def test_map_player_boxscore_columns(self):
        """
        GIVEN raw player stats DataFrame from NBA API
        WHEN _map_player_boxscore_columns is called
        THEN it renames columns to standard format
        """
        scraper = WebScrapNBAApiBoxscore(
            feature_object=self.valid_feature,
            cache_dir=Path(self.temp_dir)
        )

        raw_df = pd.DataFrame({
            'GAME_ID': ['0022400001'],
            'TEAM_ABBREVIATION': ['BOS'],
            'PLAYER_NAME': ['Jayson Tatum'],
            'PTS': [30],
            'AST': [5],
            'REB': [10]
        })

        mapped_df = scraper._map_player_boxscore_columns(raw_df, {})

        assert 'game_id' in mapped_df.columns
        assert 'tm' in mapped_df.columns
        assert 'player_name' in mapped_df.columns
        assert 'pts' in mapped_df.columns
        assert 'ast' in mapped_df.columns
        assert 'trb' in mapped_df.columns

    def test_map_team_boxscore_columns(self):
        """
        GIVEN raw team stats DataFrame from NBA API
        WHEN _map_team_boxscore_columns is called
        THEN it renames columns to standard format
        """
        scraper = WebScrapNBAApiBoxscore(
            feature_object=self.valid_feature,
            cache_dir=Path(self.temp_dir)
        )

        raw_df = pd.DataFrame({
            'GAME_ID': ['0022400001'],
            'TEAM_ABBREVIATION': ['BKN'],
            'TEAM_NAME': ['Nets'],
            'PTS': [110],
            'AST': [25]
        })

        mapped_df = scraper._map_team_boxscore_columns(raw_df, {})

        assert 'game_id' in mapped_df.columns
        assert 'tm' in mapped_df.columns
        assert mapped_df['tm'].iloc[0] == 'BKN'  # No conversion, stays as BKN

    # -------------------------------------------------------------------------
    # Integration Tests (make actual API calls)
    # -------------------------------------------------------------------------

    def test_fetch_boxscore_data_single_team(self):
        """
        GIVEN valid FeatureIn for single team boxscore
        WHEN fetch_boxscore_data is called
        THEN it returns DataFrames with expected columns
        """
        feature = FeatureIn(data_type="boxscore", season=2024, team="BOS")
        scraper = WebScrapNBAApiBoxscore(
            feature_object=feature,
            cache_dir=Path(self.temp_dir)
        )

        player_stats, team_stats = scraper.fetch_boxscore_data()

        assert isinstance(player_stats, pd.DataFrame)
        assert isinstance(team_stats, pd.DataFrame)

        if not player_stats.empty:
            assert 'id_season' in player_stats.columns
            assert 'tm' in player_stats.columns
            assert 'pts' in player_stats.columns
            assert 'player_name' in player_stats.columns


if __name__ == "__main__":
    import unittest
    unittest.main()