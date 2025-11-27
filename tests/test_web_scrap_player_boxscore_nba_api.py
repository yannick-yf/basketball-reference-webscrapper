"""Test NBABoxscoreExtractor"""

import json
import os
import tempfile
from pathlib import Path
from unittest import TestCase

import pandas as pd

from basketball_reference_webscrapper.web_scrap_player_boxscore_nba_api import (
    FailedGameRecord,
    NBABoxscoreExtractor,
)


class TestNBABoxscoreExtractor(TestCase):
    """Tests for the NBABoxscoreExtractor class that fetches boxscore data from NBA API."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = NBABoxscoreExtractor(
            base_delay=0.6,
            rate_limit_wait=300.0,
            batch_size=500,
            batch_cooldown=60.0,
            cache_dir=Path(self.temp_dir)
        )

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        # Clean up temp directory
        for file in Path(self.temp_dir).glob("*"):
            file.unlink()
        Path(self.temp_dir).rmdir()

    # -------------------------------------------------------------------------
    # Initialization Tests
    # -------------------------------------------------------------------------

    def test_initialization(self):
        """
        GIVEN valid constructor arguments
        WHEN NBABoxscoreExtractor is initialized
        THEN it creates an instance with correct attributes
        """
        extractor = NBABoxscoreExtractor(
            base_delay=0.5,
            batch_size=400,
            cache_dir=Path(self.temp_dir)
        )

        assert extractor is not None
        assert extractor.base_delay == 0.5
        assert extractor.batch_size == 400
        assert extractor._request_count == 0
        assert len(extractor._fetched_game_ids) == 0
        assert len(extractor._failed_games) == 0

    def test_initialization_creates_cache_dir(self):
        """
        GIVEN a non-existent cache directory path
        WHEN NBABoxscoreExtractor is initialized
        THEN it creates the cache directory
        """
        new_cache_dir = Path(self.temp_dir) / "new_cache"
        assert not new_cache_dir.exists()

        extractor = NBABoxscoreExtractor(cache_dir=new_cache_dir)

        assert new_cache_dir.exists()
        new_cache_dir.rmdir()

    # -------------------------------------------------------------------------
    # get_nba_team_abbreviations Tests
    # -------------------------------------------------------------------------

    def test_get_nba_team_abbreviations_returns_30_teams(self):
        """
        GIVEN an NBABoxscoreExtractor instance
        WHEN get_nba_team_abbreviations is called
        THEN it returns exactly 30 NBA team abbreviations
        """
        teams = self.extractor.get_nba_team_abbreviations()

        assert len(teams) == 30

    def test_get_nba_team_abbreviations_contains_known_teams(self):
        """
        GIVEN an NBABoxscoreExtractor instance
        WHEN get_nba_team_abbreviations is called
        THEN it contains known NBA team abbreviations
        """
        teams = self.extractor.get_nba_team_abbreviations()

        known_teams = ["BOS", "LAL", "GSW", "MIA", "CHI", "NYK"]
        for team in known_teams:
            assert team in teams, f"Expected {team} to be in team list"

    # -------------------------------------------------------------------------
    # _get_team_id_from_abbr Tests
    # -------------------------------------------------------------------------

    def test_get_team_id_from_abbr_valid(self):
        """
        GIVEN a valid team abbreviation
        WHEN _get_team_id_from_abbr is called
        THEN it returns the correct team ID
        """
        bos_id = self.extractor._get_team_id_from_abbr("BOS")
        lal_id = self.extractor._get_team_id_from_abbr("LAL")

        assert bos_id == 1610612738
        assert lal_id == 1610612747

    def test_get_team_id_from_abbr_invalid(self):
        """
        GIVEN an invalid team abbreviation
        WHEN _get_team_id_from_abbr is called
        THEN it returns None
        """
        invalid_id = self.extractor._get_team_id_from_abbr("INVALID")
        gleague_id = self.extractor._get_team_id_from_abbr("SLC")  # G-League team

        assert invalid_id is None
        assert gleague_id is None

    # -------------------------------------------------------------------------
    # _is_rate_limit_error Tests
    # -------------------------------------------------------------------------

    def test_is_rate_limit_error_detects_429(self):
        """
        GIVEN an exception with rate limit indicators
        WHEN _is_rate_limit_error is called
        THEN it returns True
        """
        error_429 = Exception("HTTP Error 429: Too Many Requests")
        error_timeout = Exception("Connection timed out")
        error_reset = Exception("Connection reset by peer")

        assert self.extractor._is_rate_limit_error(error_429) is True
        assert self.extractor._is_rate_limit_error(error_timeout) is True
        assert self.extractor._is_rate_limit_error(error_reset) is True

    def test_is_rate_limit_error_returns_false_for_other_errors(self):
        """
        GIVEN an exception without rate limit indicators
        WHEN _is_rate_limit_error is called
        THEN it returns False
        """
        error_value = ValueError("Invalid game ID format")
        error_key = KeyError("GAME_ID column not found")

        assert self.extractor._is_rate_limit_error(error_value) is False
        assert self.extractor._is_rate_limit_error(error_key) is False

    # -------------------------------------------------------------------------
    # _record_failed_game Tests
    # -------------------------------------------------------------------------

    def test_record_failed_game_adds_to_dict(self):
        """
        GIVEN a game ID and failure reason
        WHEN _record_failed_game is called
        THEN it adds the game to _failed_games dict
        """
        self.extractor._record_failed_game("0022400123", "Connection timeout")

        assert "0022400123" in self.extractor._failed_games
        record = self.extractor._failed_games["0022400123"]
        assert record.game_id == "0022400123"
        assert record.failure_reason == "Connection timeout"
        assert record.timestamp is not None

    def test_record_failed_game_overwrites_existing(self):
        """
        GIVEN a game ID that already failed
        WHEN _record_failed_game is called again
        THEN it overwrites the previous record
        """
        self.extractor._record_failed_game("0022400123", "First error")
        self.extractor._record_failed_game("0022400123", "Second error")

        assert len(self.extractor._failed_games) == 1
        assert self.extractor._failed_games["0022400123"].failure_reason == "Second error"

    # -------------------------------------------------------------------------
    # get_failed_games_summary Tests
    # -------------------------------------------------------------------------

    def test_get_failed_games_summary_empty(self):
        """
        GIVEN no failed games
        WHEN get_failed_games_summary is called
        THEN it returns empty DataFrame with correct columns
        """
        summary = self.extractor.get_failed_games_summary()

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 0
        assert list(summary.columns) == ["game_id", "failure_reason", "timestamp"]

    def test_get_failed_games_summary_with_failures(self):
        """
        GIVEN some failed games
        WHEN get_failed_games_summary is called
        THEN it returns DataFrame with all failed games
        """
        self.extractor._record_failed_game("0022400001", "Error 1")
        self.extractor._record_failed_game("0022400002", "Error 2")

        summary = self.extractor.get_failed_games_summary()

        assert len(summary) == 2
        assert set(summary["game_id"]) == {"0022400001", "0022400002"}

    # -------------------------------------------------------------------------
    # save_failed_games_cache / load_failed_games_cache Tests
    # -------------------------------------------------------------------------

    def test_save_failed_games_cache_creates_file(self):
        """
        GIVEN failed games in the extractor
        WHEN save_failed_games_cache is called
        THEN it creates a JSON file with the failed games
        """
        self.extractor._record_failed_game("0022400001", "Test error")

        filepath = self.extractor.save_failed_games_cache("test_cache.json")

        assert Path(filepath).exists()
        with open(filepath, "r") as f:
            data = json.load(f)
        assert "0022400001" in data

    def test_load_failed_games_cache_restores_games(self):
        """
        GIVEN a saved cache file
        WHEN load_failed_games_cache is called
        THEN it restores the failed games to the extractor
        """
        # Create and save cache
        self.extractor._record_failed_game("0022400001", "Test error")
        filepath = self.extractor.save_failed_games_cache("test_cache.json")

        # Create new extractor and load cache
        new_extractor = NBABoxscoreExtractor(cache_dir=Path(self.temp_dir))
        count = new_extractor.load_failed_games_cache(filepath)

        assert count == 1
        assert "0022400001" in new_extractor._failed_games

    def test_save_failed_games_cache_empty_returns_empty_string(self):
        """
        GIVEN no failed games
        WHEN save_failed_games_cache is called
        THEN it returns empty string and creates no file
        """
        filepath = self.extractor.save_failed_games_cache("test_cache.json")

        assert filepath == ""

    # -------------------------------------------------------------------------
    # get_fetched_game_ids Tests
    # -------------------------------------------------------------------------

    def test_get_fetched_game_ids_returns_copy(self):
        """
        GIVEN some fetched game IDs
        WHEN get_fetched_game_ids is called
        THEN it returns a copy of the set (not the original)
        """
        self.extractor._fetched_game_ids.add("0022400001")
        self.extractor._fetched_game_ids.add("0022400002")

        fetched = self.extractor.get_fetched_game_ids()

        assert fetched == {"0022400001", "0022400002"}
        # Verify it's a copy
        fetched.add("0022400003")
        assert "0022400003" not in self.extractor._fetched_game_ids

    # -------------------------------------------------------------------------
    # get_request_count Tests
    # -------------------------------------------------------------------------

    def test_get_request_count_initial(self):
        """
        GIVEN a freshly initialized extractor
        WHEN get_request_count is called
        THEN it returns 0
        """
        count = self.extractor.get_request_count()

        assert count == 0

    def test_get_request_count_after_increment(self):
        """
        GIVEN an extractor with incremented request count
        WHEN get_request_count is called
        THEN it returns the correct count
        """
        self.extractor._request_count = 42

        count = self.extractor.get_request_count()

        assert count == 42

    # -------------------------------------------------------------------------
    # _check_and_reset_session Tests
    # -------------------------------------------------------------------------

    def test_check_and_reset_session_increments_count(self):
        """
        GIVEN an extractor with request count below batch_size
        WHEN _check_and_reset_session is called
        THEN it increments the request count
        """
        initial_count = self.extractor._request_count

        self.extractor._check_and_reset_session()

        assert self.extractor._request_count == initial_count + 1

    def test_check_and_reset_session_resets_at_batch_size(self):
        """
        GIVEN an extractor with request count at batch_size - 1
        WHEN _check_and_reset_session is called
        THEN it resets the request count to 0 after incrementing
        """
        # Use small batch size for testing
        extractor = NBABoxscoreExtractor(
            batch_size=5,
            batch_cooldown=0.1,  # Short cooldown for test
            cache_dir=Path(self.temp_dir)
        )
        extractor._request_count = 4  # One less than batch_size

        extractor._check_and_reset_session()

        # After increment to 5, should trigger reset to 0
        assert extractor._request_count == 0

    # -------------------------------------------------------------------------
    # _reset_session Tests
    # -------------------------------------------------------------------------

    def test_reset_session_resets_request_count(self):
        """
        GIVEN an extractor with non-zero request count
        WHEN _reset_session is called
        THEN it resets the request count to 0
        """
        self.extractor._request_count = 100

        self.extractor._reset_session()

        assert self.extractor._request_count == 0

    def test_reset_session_preserves_fetched_game_ids(self):
        """
        GIVEN an extractor with fetched game IDs
        WHEN _reset_session is called
        THEN it preserves the fetched game IDs
        """
        self.extractor._fetched_game_ids.add("0022400001")
        self.extractor._fetched_game_ids.add("0022400002")

        self.extractor._reset_session()

        assert "0022400001" in self.extractor._fetched_game_ids
        assert "0022400002" in self.extractor._fetched_game_ids

    def test_reset_session_preserves_failed_games(self):
        """
        GIVEN an extractor with failed games
        WHEN _reset_session is called
        THEN it preserves the failed games
        """
        self.extractor._record_failed_game("0022400001", "Test error")

        self.extractor._reset_session()

        assert "0022400001" in self.extractor._failed_games

    # -------------------------------------------------------------------------
    # Integration Tests (make actual API calls)
    # -------------------------------------------------------------------------

    def test_get_team_games_single_team(self):
        """
        GIVEN a valid team abbreviation
        WHEN get_team_games is called
        THEN it returns DataFrame with game information
        """
        games = self.extractor.get_team_games(
            team_abbr="BOS",
            season="2024-25",
            season_type="Regular Season"
        )

        assert isinstance(games, pd.DataFrame)
        assert "GAME_ID" in games.columns
        assert "TEAM_ABBREVIATION" in games.columns
        assert len(games) > 0

    def test_get_team_games_invalid_team_raises(self):
        """
        GIVEN an invalid team abbreviation
        WHEN get_team_games is called
        THEN it raises ValueError
        """
        with self.assertRaises(ValueError) as context:
            self.extractor.get_team_games(team_abbr="INVALID", season="2024-25")

        assert "Invalid team abbreviation" in str(context.exception)

    def test_extract_all_boxscores_limited(self):
        """
        GIVEN valid season parameters
        WHEN extract_all_boxscores is called with max_games limit
        THEN it returns player and team stats DataFrames
        """
        player_stats, team_stats = self.extractor.extract_all_boxscores(
            season="2024-25",
            season_type="Regular Season",
            team_abbr="BOS",
            max_games=2
        )

        assert isinstance(player_stats, pd.DataFrame)
        assert isinstance(team_stats, pd.DataFrame)

        if not player_stats.empty:
            assert "PLAYER_NAME" in player_stats.columns
            assert "PTS" in player_stats.columns
            assert len(self.extractor.get_fetched_game_ids()) <= 2


if __name__ == "__main__":
    import unittest
    unittest.main()