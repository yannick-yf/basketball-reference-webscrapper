"""
Utility functions for NBA team validation and season-based filtering.

This module provides shared functionality for validating team abbreviations
against their historically valid seasons, handling cases like:
- CHO vs CHA (Charlotte Hornets vs Bobcats)
- BRK vs NJN (Brooklyn vs New Jersey Nets)
- OKC vs SEA (Oklahoma City vs Seattle SuperSonics)

Used by both WebScrapNBAApi and WebScrapNBAApiBoxscore classes.
"""

from typing import Dict, List, Optional
import importlib_resources
import yaml

from basketball_reference_webscrapper.utils.logs import get_logger

__all__ = [
    'load_team_season_validity',
    'is_team_valid_for_season',
    'filter_valid_teams_for_season'
]

logger = get_logger("TEAM_UTILS", log_level="INFO")

# Module-level cache to avoid repeated file reads
_team_season_validity_cache: Optional[Dict[str, Dict]] = None

# Default minimum supported season
MIN_SUPPORTED_SEASON = 2000


def load_team_season_validity() -> Dict[str, Dict]:
    """
    Load and cache team season validity configuration from YAML file.

    The configuration maps team abbreviations to their valid season ranges,
    handling historical team name changes and relocations.

    Returns:
        Dict[str, Dict]: Dictionary mapping team abbreviations to their
            valid season ranges with 'start_season' and 'end_season' keys.
            Returns empty dict if file cannot be loaded.

    Example:
        >>> config = load_team_season_validity()
        >>> config['CHO']
        {'start_season': 2015, 'end_season': None}
    """
    global _team_season_validity_cache

    if _team_season_validity_cache is not None:
        return _team_season_validity_cache

    try:
        ref = (
            importlib_resources.files("basketball_reference_webscrapper")
            / "constants/team_season_validity.yaml"
        )
        with importlib_resources.as_file(ref) as path:
            with open(path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
                _team_season_validity_cache = config.get("team_season_validity", {})
                logger.debug(
                    "Loaded team season validity config with %d teams",
                    len(_team_season_validity_cache)
                )
                return _team_season_validity_cache
    except FileNotFoundError:
        logger.warning(
            "team_season_validity.yaml not found. "
            "All teams will be considered valid for all seasons."
        )
        _team_season_validity_cache = {}
        return _team_season_validity_cache
    except Exception as e:
        logger.warning(
            "Could not load team_season_validity.yaml: %s. "
            "All teams will be considered valid for all seasons.",
            str(e)
        )
        _team_season_validity_cache = {}
        return _team_season_validity_cache


def is_team_valid_for_season(
    team_abbr: str,
    season: int,
    min_supported_season: int = MIN_SUPPORTED_SEASON
) -> bool:
    """
    Check if a team abbreviation is valid for a given season.

    This handles historical team name changes. For example:
    - CHO (Charlotte Hornets) is valid from 2015 onwards
    - CHA (Charlotte Bobcats) is valid from 2005-2014
    - Both map to the same team_id, but only one is valid per season

    Args:
        team_abbr (str): Team abbreviation (e.g., 'CHO', 'CHA', 'BOS').
        season (int): Season year (e.g., 2024 for the 2023-24 season).
        min_supported_season (int): Minimum supported season year.
            Defaults to 2000.

    Returns:
        bool: True if team is valid for the season, False otherwise.
            Returns True if team is not in config (backward compatibility).

    Example:
        >>> is_team_valid_for_season('CHO', 2024)
        True
        >>> is_team_valid_for_season('CHA', 2024)
        False
        >>> is_team_valid_for_season('CHA', 2010)
        True
    """
    validity_config = load_team_season_validity()

    # If no validity config loaded, assume all teams are valid
    if not validity_config:
        return True

    # If team not in config, assume it's valid (backward compatibility)
    if team_abbr not in validity_config:
        return True

    validity = validity_config[team_abbr]
    start_season = validity.get("start_season", min_supported_season)
    end_season = validity.get("end_season")  # None means current/ongoing

    # Check if season falls within valid range
    if season < start_season:
        return False

    if end_season is not None and season > end_season:
        return False

    return True


def filter_valid_teams_for_season(
    team_list: List[str],
    season: int,
    min_supported_season: int = MIN_SUPPORTED_SEASON
) -> List[str]:
    """
    Filter a list of team abbreviations to only those valid for a season.

    This is a convenience function that applies is_team_valid_for_season
    to a list of teams.

    Args:
        team_list (List[str]): List of team abbreviations to filter.
        season (int): Season year (e.g., 2024 for the 2023-24 season).
        min_supported_season (int): Minimum supported season year.
            Defaults to 2000.

    Returns:
        List[str]: Filtered list containing only team abbreviations
            that are valid for the specified season.

    Example:
        >>> filter_valid_teams_for_season(['CHO', 'CHA', 'BOS'], 2024)
        ['CHO', 'BOS']
        >>> filter_valid_teams_for_season(['CHO', 'CHA', 'BOS'], 2010)
        ['CHA', 'BOS']
    """
    return [
        team for team in team_list
        if is_team_valid_for_season(team, season, min_supported_season)
    ]


def get_excluded_teams_for_season(
    team_list: List[str],
    season: int
) -> List[str]:
    """
    Get list of teams that are NOT valid for a given season.

    Useful for logging which teams were filtered out.

    Args:
        team_list (List[str]): List of team abbreviations to check.
        season (int): Season year.

    Returns:
        List[str]: List of team abbreviations that are invalid for the season.

    Example:
        >>> get_excluded_teams_for_season(['CHO', 'CHA', 'BOS'], 2024)
        ['CHA']
    """
    return [
        team for team in team_list
        if not is_team_valid_for_season(team, season)
    ]


def clear_cache() -> None:
    """
    Clear the cached team season validity configuration.

    Useful for testing or when the configuration file has been updated.
    """
    global _team_season_validity_cache
    _team_season_validity_cache = None
    logger.debug("Team season validity cache cleared")