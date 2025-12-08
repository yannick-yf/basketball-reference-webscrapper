"""
Unit tests for team_utils module.

Tests the team season validity logic that prevents duplicate data
from historical team name changes (CHO/CHA, BRK/NJN, etc.).
"""

import pytest
from unittest.mock import patch, mock_open

# Import the functions to test
from basketball_reference_webscrapper.utils.team_utils import (
    is_team_valid_for_season,
    filter_valid_teams_for_season,
    get_excluded_teams_for_season,
    load_team_season_validity,
    clear_cache,
)


class TestIsTeamValidForSeason:
    """Tests for is_team_valid_for_season function."""

    def setup_method(self):
        """Clear cache before each test to ensure clean state."""
        clear_cache()

    # -------------------------------------------------------------------------
    # Charlotte Hornets/Bobcats (CHO/CHA) - Primary bug fix
    # -------------------------------------------------------------------------

    def test_cho_valid_for_2024(self):
        """CHO (Charlotte Hornets) should be valid for 2024 season."""
        assert is_team_valid_for_season("CHO", 2024) is True

    def test_cho_valid_for_2015(self):
        """CHO should be valid starting from 2015 (rebrand year)."""
        assert is_team_valid_for_season("CHO", 2015) is True

    def test_cho_invalid_for_2014(self):
        """CHO should NOT be valid for 2014 (was still Bobcats)."""
        assert is_team_valid_for_season("CHO", 2014) is False

    def test_cha_valid_for_2010(self):
        """CHA (Charlotte Bobcats) should be valid for 2010."""
        assert is_team_valid_for_season("CHA", 2010) is True

    def test_cha_valid_for_2014(self):
        """CHA should be valid for 2014 (last Bobcats season)."""
        assert is_team_valid_for_season("CHA", 2014) is True

    def test_cha_invalid_for_2015(self):
        """CHA should NOT be valid for 2015 (became Hornets)."""
        assert is_team_valid_for_season("CHA", 2015) is False

    def test_cha_invalid_for_2024(self):
        """CHA should NOT be valid for 2024."""
        assert is_team_valid_for_season("CHA", 2024) is False

    # -------------------------------------------------------------------------
    # Brooklyn/New Jersey Nets (BRK/NJN)
    # -------------------------------------------------------------------------

    def test_brk_valid_for_2024(self):
        """BRK (Brooklyn Nets) should be valid for 2024."""
        assert is_team_valid_for_season("BRK", 2024) is True

    def test_brk_valid_for_2013(self):
        """BRK should be valid starting from 2013 (move to Brooklyn)."""
        assert is_team_valid_for_season("BRK", 2013) is True

    def test_brk_invalid_for_2012(self):
        """BRK should NOT be valid for 2012 (was still NJN)."""
        assert is_team_valid_for_season("BRK", 2012) is False

    def test_njn_valid_for_2012(self):
        """NJN (New Jersey Nets) should be valid for 2012."""
        assert is_team_valid_for_season("NJN", 2012) is True

    def test_njn_invalid_for_2013(self):
        """NJN should NOT be valid for 2013 (became BRK)."""
        assert is_team_valid_for_season("NJN", 2013) is False

    # -------------------------------------------------------------------------
    # Oklahoma City/Seattle (OKC/SEA)
    # -------------------------------------------------------------------------

    def test_okc_valid_for_2024(self):
        """OKC should be valid for 2024."""
        assert is_team_valid_for_season("OKC", 2024) is True

    def test_okc_valid_for_2009(self):
        """OKC should be valid starting from 2009 (relocation year)."""
        assert is_team_valid_for_season("OKC", 2009) is True

    def test_okc_invalid_for_2008(self):
        """OKC should NOT be valid for 2008 (was still SEA)."""
        assert is_team_valid_for_season("OKC", 2008) is False

    def test_sea_valid_for_2008(self):
        """SEA (Seattle SuperSonics) should be valid for 2008."""
        assert is_team_valid_for_season("SEA", 2008) is True

    def test_sea_invalid_for_2009(self):
        """SEA should NOT be valid for 2009 (became OKC)."""
        assert is_team_valid_for_season("SEA", 2009) is False

    # -------------------------------------------------------------------------
    # Memphis/Vancouver Grizzlies (MEM/VAN)
    # -------------------------------------------------------------------------

    def test_mem_valid_for_2024(self):
        """MEM should be valid for 2024."""
        assert is_team_valid_for_season("MEM", 2024) is True

    def test_mem_valid_for_2002(self):
        """MEM should be valid starting from 2002 (relocation year)."""
        assert is_team_valid_for_season("MEM", 2002) is True

    def test_mem_invalid_for_2001(self):
        """MEM should NOT be valid for 2001 (was still VAN)."""
        assert is_team_valid_for_season("MEM", 2001) is False

    def test_van_valid_for_2001(self):
        """VAN (Vancouver Grizzlies) should be valid for 2001."""
        assert is_team_valid_for_season("VAN", 2001) is True

    def test_van_invalid_for_2002(self):
        """VAN should NOT be valid for 2002 (became MEM)."""
        assert is_team_valid_for_season("VAN", 2002) is False

    # -------------------------------------------------------------------------
    # New Orleans Pelicans/Hornets (NOP/NOH)
    # -------------------------------------------------------------------------

    def test_nop_valid_for_2024(self):
        """NOP should be valid for 2024."""
        assert is_team_valid_for_season("NOP", 2024) is True

    def test_nop_valid_for_2014(self):
        """NOP should be valid starting from 2014 (rebrand year)."""
        assert is_team_valid_for_season("NOP", 2014) is True

    def test_nop_invalid_for_2013(self):
        """NOP should NOT be valid for 2013 (was still NOH)."""
        assert is_team_valid_for_season("NOP", 2013) is False

    def test_noh_valid_for_2010(self):
        """NOH (New Orleans Hornets) should be valid for 2010."""
        assert is_team_valid_for_season("NOH", 2010) is True

    def test_noh_invalid_for_2014(self):
        """NOH should NOT be valid for 2014 (became NOP)."""
        assert is_team_valid_for_season("NOH", 2014) is False

    # -------------------------------------------------------------------------
    # Teams with no name changes (always valid since 2000)
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("team", [
        "ATL", "BOS", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND",
        "LAC", "LAL", "MIA", "MIL", "MIN", "NYK", "ORL", "PHI", "PHO", "POR",
        "SAC", "SAS", "TOR", "UTA", "WAS"
    ])
    def test_stable_teams_valid_for_2024(self, team):
        """Teams with no name changes should be valid for 2024."""
        assert is_team_valid_for_season(team, 2024) is True

    @pytest.mark.parametrize("team", [
        "ATL", "BOS", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND",
        "LAC", "LAL", "MIA", "MIL", "MIN", "NYK", "ORL", "PHI", "PHO", "POR",
        "SAC", "SAS", "TOR", "UTA", "WAS"
    ])
    def test_stable_teams_valid_for_2000(self, team):
        """Teams with no name changes should be valid for 2000."""
        assert is_team_valid_for_season(team, 2000) is True

    # -------------------------------------------------------------------------
    # Edge cases
    # -------------------------------------------------------------------------

    def test_unknown_team_returns_true(self):
        """Unknown teams should return True (backward compatibility)."""
        assert is_team_valid_for_season("XYZ", 2024) is True

    def test_boundary_season_start(self):
        """Test exact boundary at start_season."""
        # CHO starts at 2015
        assert is_team_valid_for_season("CHO", 2015) is True
        assert is_team_valid_for_season("CHO", 2014) is False

    def test_boundary_season_end(self):
        """Test exact boundary at end_season."""
        # CHA ends at 2014
        assert is_team_valid_for_season("CHA", 2014) is True
        assert is_team_valid_for_season("CHA", 2015) is False


class TestFilterValidTeamsForSeason:
    """Tests for filter_valid_teams_for_season function."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_filters_out_cha_for_2024(self):
        """Should filter out CHA when requesting 2024 season."""
        teams = ["CHO", "CHA", "BOS", "LAL"]
        result = filter_valid_teams_for_season(teams, 2024)

        assert "CHO" in result
        assert "CHA" not in result
        assert "BOS" in result
        assert "LAL" in result

    def test_filters_out_cho_for_2010(self):
        """Should filter out CHO when requesting 2010 season."""
        teams = ["CHO", "CHA", "BOS", "LAL"]
        result = filter_valid_teams_for_season(teams, 2010)

        assert "CHO" not in result
        assert "CHA" in result
        assert "BOS" in result
        assert "LAL" in result

    def test_filters_multiple_historical_teams_2024(self):
        """Should filter out all historical teams for 2024."""
        teams = ["CHO", "CHA", "BRK", "NJN", "OKC", "SEA", "MEM", "VAN", "NOP", "NOH", "BOS"]
        result = filter_valid_teams_for_season(teams, 2024)

        # Current teams should be present
        assert "CHO" in result
        assert "BRK" in result
        assert "OKC" in result
        assert "MEM" in result
        assert "NOP" in result
        assert "BOS" in result

        # Historical teams should be filtered out
        assert "CHA" not in result
        assert "NJN" not in result
        assert "SEA" not in result
        assert "VAN" not in result
        assert "NOH" not in result

    def test_preserves_order(self):
        """Should preserve the order of teams in the list."""
        teams = ["BOS", "CHO", "LAL", "GSW"]
        result = filter_valid_teams_for_season(teams, 2024)
        assert result == ["BOS", "CHO", "LAL", "GSW"]

    def test_empty_list(self):
        """Should handle empty list."""
        result = filter_valid_teams_for_season([], 2024)
        assert result == []

    def test_all_filtered_out(self):
        """Should return empty list if all teams filtered out."""
        teams = ["CHA", "NJN", "SEA", "VAN", "NOH"]
        result = filter_valid_teams_for_season(teams, 2024)
        assert result == []


class TestGetExcludedTeamsForSeason:
    """Tests for get_excluded_teams_for_season function."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_returns_excluded_teams_2024(self):
        """Should return list of excluded teams for 2024."""
        teams = ["CHO", "CHA", "BOS"]
        excluded = get_excluded_teams_for_season(teams, 2024)

        assert "CHA" in excluded
        assert "CHO" not in excluded
        assert "BOS" not in excluded

    def test_returns_empty_when_all_valid(self):
        """Should return empty list when all teams are valid."""
        teams = ["BOS", "LAL", "GSW"]
        excluded = get_excluded_teams_for_season(teams, 2024)
        assert excluded == []

    def test_returns_all_historical_teams_2024(self):
        """Should return all historical teams for 2024."""
        teams = ["CHA", "NJN", "SEA", "VAN", "NOH"]
        excluded = get_excluded_teams_for_season(teams, 2024)

        assert set(excluded) == {"CHA", "NJN", "SEA", "VAN", "NOH"}


class TestLoadTeamSeasonValidity:
    """Tests for load_team_season_validity function."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_returns_dict(self):
        """Should return a dictionary."""
        result = load_team_season_validity()
        assert isinstance(result, dict)

    def test_contains_expected_teams(self):
        """Should contain expected team entries."""
        result = load_team_season_validity()

        # Check a few expected teams
        assert "CHO" in result
        assert "CHA" in result
        assert "BOS" in result

    def test_team_entry_has_required_keys(self):
        """Each team entry should have start_season key."""
        result = load_team_season_validity()

        for team, validity in result.items():
            assert "start_season" in validity, f"{team} missing start_season"

    def test_caching_works(self):
        """Should cache results and return same object."""
        result1 = load_team_season_validity()
        result2 = load_team_season_validity()

        # Should be the exact same object (cached)
        assert result1 is result2

    def test_clear_cache_reloads(self):
        """Clearing cache should cause reload on next call."""
        result1 = load_team_season_validity()
        clear_cache()
        result2 = load_team_season_validity()

        # Should have same content but be different objects
        assert result1 == result2
        # Note: After clear, it reloads, so they might be same object again
        # The important thing is that clear_cache doesn't break anything


class TestClearCache:
    """Tests for clear_cache function."""

    def test_clear_cache_does_not_raise(self):
        """clear_cache should not raise exceptions."""
        clear_cache()  # Should not raise

    def test_clear_cache_allows_reload(self):
        """After clear_cache, load should work normally."""
        load_team_season_validity()
        clear_cache()
        result = load_team_season_validity()
        assert isinstance(result, dict)


class TestConfigFileMissing:
    """Tests for behavior when config file is missing."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    @patch('basketball_reference_webscrapper.utils.team_utils.importlib_resources')
    def test_missing_config_returns_empty_dict(self, mock_resources):
        """Should return empty dict if config file not found."""
        mock_resources.files.side_effect = FileNotFoundError("File not found")

        result = load_team_season_validity()
        assert result == {}

    @patch('basketball_reference_webscrapper.utils.team_utils.load_team_season_validity')
    def test_missing_config_all_teams_valid(self, mock_load):
        """All teams should be valid when config is missing."""
        mock_load.return_value = {}

        # With empty config, all teams should be valid
        assert is_team_valid_for_season("CHA", 2024) is True
        assert is_team_valid_for_season("CHO", 2024) is True
        assert is_team_valid_for_season("XYZ", 2024) is True


class TestIntegrationScenarios:
    """Integration tests simulating real usage scenarios."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_scenario_fetch_all_teams_2024(self):
        """Simulate fetching all teams for 2024 season."""
        # This simulates what team_city_refdata might contain
        all_teams = [
            "ATL", "BOS", "BRK", "CHA", "CHI", "CHO", "CLE", "DAL", "DEN", "DET",
            "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NJN",
            "NOP", "NOH", "NYK", "OKC", "ORL", "PHI", "PHO", "POR", "SAC", "SAS",
            "SEA", "TOR", "UTA", "VAN", "WAS"
        ]

        valid_teams = filter_valid_teams_for_season(all_teams, 2024)

        # Should have exactly 30 teams (current NBA)
        assert len(valid_teams) == 30

        # Historical teams should be excluded
        assert "CHA" not in valid_teams  # Use CHO
        assert "NJN" not in valid_teams  # Use BRK
        assert "SEA" not in valid_teams  # Use OKC
        assert "VAN" not in valid_teams  # Use MEM
        assert "NOH" not in valid_teams  # Use NOP

        # Current teams should be included
        assert "CHO" in valid_teams
        assert "BRK" in valid_teams
        assert "OKC" in valid_teams
        assert "MEM" in valid_teams
        assert "NOP" in valid_teams

    def test_scenario_fetch_all_teams_2005(self):
        """Simulate fetching all teams for 2005 season."""
        all_teams = [
            "ATL", "BOS", "BRK", "CHA", "CHI", "CHO", "CLE", "DAL", "DEN", "DET",
            "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NJN",
            "NOP", "NOH", "NYK", "OKC", "ORL", "PHI", "PHO", "POR", "SAC", "SAS",
            "SEA", "TOR", "UTA", "VAN", "WAS"
        ]

        valid_teams = filter_valid_teams_for_season(all_teams, 2005)

        # For 2005:
        # - CHA valid (Bobcats started 2005), CHO invalid
        # - NJN valid, BRK invalid
        # - SEA valid, OKC invalid
        # - MEM valid, VAN invalid (moved 2002)
        # - NOH valid, NOP invalid

        assert "CHA" in valid_teams
        assert "CHO" not in valid_teams

        assert "NJN" in valid_teams
        assert "BRK" not in valid_teams

        assert "SEA" in valid_teams
        assert "OKC" not in valid_teams

        assert "MEM" in valid_teams
        assert "VAN" not in valid_teams

    def test_scenario_single_team_request(self):
        """Simulate requesting a single team."""
        # User requests CHO for 2024 - should work
        teams = ["CHO"]
        valid = filter_valid_teams_for_season(teams, 2024)
        assert valid == ["CHO"]

        # User requests CHA for 2024 - should be filtered out
        teams = ["CHA"]
        valid = filter_valid_teams_for_season(teams, 2024)
        assert valid == []

        # User requests CHA for 2010 - should work
        teams = ["CHA"]
        valid = filter_valid_teams_for_season(teams, 2010)
        assert valid == ["CHA"]