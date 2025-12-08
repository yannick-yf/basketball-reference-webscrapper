"""
Class that fetches NBA player and team boxscore data from the official NBA API
(stats.nba.com) using the nba_api Python package.

This module provides robust NBA boxscore extraction with team-by-team iteration,
game ID deduplication, session management, and retry capabilities.
"""

# Standard library
import gc
import importlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Third-party
import pandas as pd
import importlib_resources
import yaml
from nba_api.stats.endpoints import boxscoretraditionalv2, leaguegamefinder, boxscoretraditionalv3
from nba_api.stats.static import teams

# Local
from basketball_reference_webscrapper.data_models.feature_model import FeatureIn
from basketball_reference_webscrapper.utils.logs import get_logger
from basketball_reference_webscrapper.utils.team_utils import (
    is_team_valid_for_season,
    get_excluded_teams_for_season
)

__all__ = ['WebScrapNBAApiBoxscore', 'FailedGameRecord']

logger = get_logger("SCRAPER_NBA_API_BOXSCORE", log_level="INFO")

# Constants
VALID_DATA_TYPES = ["boxscore", "boxscore_player", "boxscore_team"]
MIN_SUPPORTED_SEASON = 2000
DEFAULT_BASE_DELAY = 0.6  # seconds between requests
DEFAULT_BATCH_SIZE = 500  # requests before session reset
DEFAULT_BATCH_COOLDOWN = 60.0  # seconds between batches

# NBA Teams only (excludes G-League)
NBA_TEAMS: List[Dict] = teams.get_teams()

# NBA Team ID mapping (current as of 2024-25 season)
# Shared with web_scrap_nba_api.py - consider extracting to constants module
NBA_TEAM_MAPPING: Dict[str, Dict] = {
    'ATL': {'team_id': 1610612737},
    'BOS': {'team_id': 1610612738},
    'BRK': {'team_id': 1610612751},
    'CHA': {'team_id': 1610612766},
    'CHI': {'team_id': 1610612741},
    'CHO': {'team_id': 1610612766},
    'CLE': {'team_id': 1610612739},
    'DAL': {'team_id': 1610612742},
    'DEN': {'team_id': 1610612743},
    'DET': {'team_id': 1610612765},
    'GSW': {'team_id': 1610612744},
    'HOU': {'team_id': 1610612745},
    'IND': {'team_id': 1610612754},
    'LAC': {'team_id': 1610612746},
    'LAL': {'team_id': 1610612747},
    'MEM': {'team_id': 1610612763},
    'MIA': {'team_id': 1610612748},
    'MIL': {'team_id': 1610612749},
    'MIN': {'team_id': 1610612750},
    'NOP': {'team_id': 1610612740},
    'NYK': {'team_id': 1610612752},
    'OKC': {'team_id': 1610612760},
    'ORL': {'team_id': 1610612753},
    'PHI': {'team_id': 1610612755},
    'PHO': {'team_id': 1610612756},
    'POR': {'team_id': 1610612757},
    'SAC': {'team_id': 1610612758},
    'SAS': {'team_id': 1610612759},
    'TOR': {'team_id': 1610612761},
    'UTA': {'team_id': 1610612762},
    'WAS': {'team_id': 1610612764},
}

# Rate limit error indicators
RATE_LIMIT_INDICATORS = [
    "429",
    "rate limit",
    "too many requests",
    "timeout",
    "timed out",
    "connection",
    "connectionerror",
    "connection reset",
    "connection refused",
    "max retries",
    "remote end closed",
    "broken pipe",
    "reset by peer"
]


@dataclass
class FailedGameRecord:
    """
    Data class representing a failed game fetch attempt.

    Attributes:
        game_id (str): NBA game ID.
        failure_reason (str): Description of the failure.
        timestamp (str): ISO format timestamp of the failure.
    """
    game_id: str
    failure_reason: str
    timestamp: str


class WebScrapNBAApiBoxscore:
    """
    Class that fetches NBA boxscore data from the official NBA API.

    This class provides robust NBA boxscore extraction with:
    - Team-by-team iteration to ensure only NBA games (no G-League)
    - Game ID deduplication to avoid redundant API calls
    - Periodic session reset to avoid connection exhaustion
    - Caching system for failed games with retry capability
    - Season-based team validity filtering to handle historical name changes

    Follows the same interface pattern as WebScrapNBAApi for consistency.

    Note:
        NBA API blocks cloud providers (AWS, Heroku, GCP, etc.).
        Use this class for local development only.

    Attributes:
        feature_object (FeatureIn): Input feature object containing data_type,
            season, and team parameters.
        base_delay (float): Delay between API calls in seconds.
        batch_size (int): Number of requests before forcing a session reset.
        batch_cooldown (float): Cooldown time in seconds between batches.
        cache_dir (Path): Directory for storing failed games cache.

    Example:
        >>> from basketball_reference_webscrapper.data_models.feature_model import FeatureIn
        >>> feature = FeatureIn(data_type='boxscore', season=2023, team='BOS')
        >>> scraper = WebScrapNBAApiBoxscore(feature_object=feature)
        >>> player_stats, team_stats = scraper.fetch_boxscore_data()
        >>> print(player_stats.head())
    """

    def __init__(
        self,
        feature_object: FeatureIn,
        base_delay: float = DEFAULT_BASE_DELAY,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_cooldown: float = DEFAULT_BATCH_COOLDOWN,
        cache_dir: Optional[Path] = None
    ) -> None:
        """
        Initialize the NBA Boxscore extractor.

        Args:
            feature_object (FeatureIn): Feature object containing data_type,
                season, and team parameters.
            base_delay (float): Delay between API calls in seconds.
                Defaults to 0.6 seconds.
            batch_size (int): Number of requests before forcing a session reset.
                Defaults to 500.
            batch_cooldown (float): Cooldown time in seconds between batches.
                Defaults to 60.0 seconds.
            cache_dir (Optional[Path]): Directory for storing failed games cache.
                Defaults to './cache'.
        """
        self.feature_object = feature_object
        self.base_delay = base_delay
        self.batch_size = batch_size
        self.batch_cooldown = batch_cooldown
        self.cache_dir = cache_dir or Path("./cache")

        # State tracking
        self._fetched_game_ids: Set[str] = set()
        self._failed_games: Dict[str, FailedGameRecord] = {}
        self._request_count: int = 0

        # Team mapping for ID lookups
        self.team_mapping = NBA_TEAM_MAPPING

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "WebScrapNBAApiBoxscore initialized with base_delay=%.2fs, batch_size=%d",
            self.base_delay,
            self.batch_size
        )

    def fetch_boxscore_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch NBA boxscore data from the official NBA API.

        This is the main entry point for fetching boxscore data. It validates
        inputs, iterates through teams and games, and returns both player
        and team statistics.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - player_stats_df: DataFrame with player-level boxscore statistics
                - team_stats_df: DataFrame with team-level boxscore statistics

        Raises:
            ValueError: If data_type, season, or team parameters are invalid.

        Example:
            >>> feature = FeatureIn(data_type='boxscore', season=2023, team='BOS')
            >>> scraper = WebScrapNBAApiBoxscore(feature_object=feature)
            >>> player_stats, team_stats = scraper.fetch_boxscore_data()
        """
        # Input validation
        self._validate_data_type()
        self._validate_season()

        # Load configuration
        config = self._load_config()

        # Load team reference data
        team_city_refdata = self._load_team_refdata()

        # Validate team input
        self._validate_team(list(team_city_refdata["team_abrev"]))

        # Filter teams based on input
        teams_to_fetch = self._filter_teams(team_city_refdata)

        # Convert season to NBA API format
        season_str = self._convert_season_format(self.feature_object.season)

        logger.info(
            "Fetching boxscore data for season %s, teams: %s",
            season_str,
            "all" if len(teams_to_fetch) == 30 else list(teams_to_fetch["team_abrev"])
        )

        # Extract boxscores
        player_stats, team_stats = self._extract_all_boxscores(
            teams_to_fetch=list(teams_to_fetch["team_abrev"]),
            season=season_str,
            season_type=self._get_season_type()
        )

        if player_stats.empty:
            logger.warning("No boxscore data was fetched. Returning empty DataFrames.")
            return pd.DataFrame(), pd.DataFrame()

        # Add standard columns
        player_stats["id_season"] = self.feature_object.season
        team_stats["id_season"] = self.feature_object.season

        # Map column names to standard format
        player_stats = self._map_player_boxscore_columns(player_stats, config)
        team_stats = self._map_team_boxscore_columns(team_stats, config)

        logger.info(
            "Successfully fetched boxscore data: %d player records, %d team records",
            len(player_stats),
            len(team_stats)
        )

        return player_stats, team_stats

    def retry_failed_games(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retry fetching boxscores for previously failed games.

        This method attempts to re-fetch boxscore data for games that failed
        during the initial extraction. Useful for recovering from transient
        API failures.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - player_stats_df: DataFrame with player stats for recovered games
                - team_stats_df: DataFrame with team stats for recovered games
        """
        if not self._failed_games:
            logger.info("No failed games to retry")
            return pd.DataFrame(), pd.DataFrame()

        failed_game_ids = list(self._failed_games.keys())
        logger.info("Retrying %d failed games", len(failed_game_ids))

        # Remove from fetched set so they can be retried
        for game_id in failed_game_ids:
            self._fetched_game_ids.discard(game_id)

        all_player_stats: List[pd.DataFrame] = []
        all_team_stats: List[pd.DataFrame] = []

        for game_id in failed_game_ids:
            logger.info("Retrying game %s", game_id)

            player_stats, team_stats = self._get_game_boxscore(game_id)

            if player_stats is not None and team_stats is not None:
                all_player_stats.append(player_stats)
                all_team_stats.append(team_stats)
                logger.info("Successfully recovered game %s", game_id)

        recovered_count = len(all_player_stats)
        logger.info(
            "Retry complete: %d/%d games recovered",
            recovered_count,
            len(failed_game_ids)
        )

        if all_player_stats:
            player_df = pd.concat(all_player_stats, ignore_index=True)
            team_df = pd.concat(all_team_stats, ignore_index=True)

            # Add season column
            player_df["id_season"] = self.feature_object.season
            team_df["id_season"] = self.feature_object.season

            return player_df, team_df

        return pd.DataFrame(), pd.DataFrame()

    def get_failed_games_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of all failed games.

        Returns:
            pd.DataFrame: Summary of failed games with columns:
                game_id, failure_reason, timestamp.
        """
        if not self._failed_games:
            return pd.DataFrame(columns=["game_id", "failure_reason", "timestamp"])

        return pd.DataFrame([
            {
                "game_id": record.game_id,
                "failure_reason": record.failure_reason,
                "timestamp": record.timestamp
            }
            for record in self._failed_games.values()
        ])

    def save_failed_games_cache(self, filename: Optional[str] = None) -> str:
        """
        Save failed games to a JSON cache file.

        Args:
            filename (Optional[str]): Output filename. If None, generates
                a timestamped filename.

        Returns:
            str: Path to the saved cache file, or empty string if no failures.
        """
        if not self._failed_games:
            logger.info("No failed games to cache")
            return ""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"failed_games_{timestamp}.json"

        filepath = self.cache_dir / filename

        cache_data = {
            game_id: {
                "game_id": record.game_id,
                "failure_reason": record.failure_reason,
                "timestamp": record.timestamp
            }
            for game_id, record in self._failed_games.items()
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)

        logger.info("Saved %d failed games to %s", len(self._failed_games), filepath)
        return str(filepath)

    def load_failed_games_cache(self, filepath: str) -> int:
        """
        Load failed games from a JSON cache file.

        Args:
            filepath (str): Path to the cache file.

        Returns:
            int: Number of failed games loaded.
        """
        cache_path = Path(filepath)

        if not cache_path.exists():
            logger.warning("Cache file not found: %s", filepath)
            return 0

        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        for game_id, data in cache_data.items():
            self._failed_games[game_id] = FailedGameRecord(
                game_id=data["game_id"],
                failure_reason=data["failure_reason"],
                timestamp=data["timestamp"]
            )

        logger.info("Loaded %d failed games from %s", len(cache_data), filepath)
        return len(cache_data)

    def get_fetched_game_ids(self) -> Set[str]:
        """
        Get set of already fetched game IDs.

        Returns:
            Set[str]: Set of game IDs that have been successfully fetched.
        """
        return self._fetched_game_ids.copy()

    def get_request_count(self) -> int:
        """
        Get current request count since last session reset.

        Returns:
            int: Number of requests made in current session.
        """
        return self._request_count

    # -------------------------------------------------------------------------
    # Private Methods - Extraction Logic
    # -------------------------------------------------------------------------

    def _extract_all_boxscores(
        self,
        teams_to_fetch: List[str],
        season: str,
        season_type: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract boxscores for all games, iterating team-by-team.

        Args:
            teams_to_fetch (List[str]): List of team abbreviations to process.
            season (str): Season in format 'YYYY-YY'.
            season_type (str): 'Regular Season', 'Playoffs', etc.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - player_stats_df: Combined player statistics
                - team_stats_df: Combined team statistics
        """
        # Reset state for fresh extraction
        self._fetched_game_ids.clear()
        self._failed_games.clear()

        logger.info("Extracting boxscores for %d team(s)", len(teams_to_fetch))

        all_player_stats: List[pd.DataFrame] = []
        all_team_stats: List[pd.DataFrame] = []

        for team_idx, team_abbr in enumerate(teams_to_fetch, 1):
            logger.info(
                "Processing team %d/%d: %s",
                team_idx,
                len(teams_to_fetch),
                team_abbr
            )

            # Get games for this team
            try:
                team_games = self._get_team_games(team_abbr, season, season_type)
            except Exception as e:
                logger.error(
                    "Failed to get games for team %s: %s",
                    team_abbr,
                    str(e)
                )
                continue

            if team_games.empty:
                logger.warning("No games found for team %s", team_abbr)
                continue

            # Process each game
            game_ids = team_games["GAME_ID"].unique()

            for game_id in game_ids:
                # Skip already fetched
                if game_id in self._fetched_game_ids:
                    continue

                player_stats, team_stats = self._get_game_boxscore(game_id)

                if player_stats is not None and team_stats is not None:
                    all_player_stats.append(player_stats)
                    all_team_stats.append(team_stats)

                    if len(self._fetched_game_ids) % 50 == 0:
                        logger.info(
                            "Progress: %d games processed",
                            len(self._fetched_game_ids)
                        )

        success_count = len(all_player_stats)
        fail_count = len(self._failed_games)

        logger.info(
            "Extraction complete: %d succeeded, %d failed",
            success_count,
            fail_count
        )

        if all_player_stats:
            combined_player = pd.concat(all_player_stats, ignore_index=True)
            combined_team = pd.concat(all_team_stats, ignore_index=True)
            return combined_player, combined_team

        return pd.DataFrame(), pd.DataFrame()

    def _get_team_games(
        self,
        team_abbr: str,
        season: str,
        season_type: str
    ) -> pd.DataFrame:
        """
        Get all games for a specific team.

        Args:
            team_abbr (str): Team abbreviation.
            season (str): Season in format 'YYYY-YY'.
            season_type (str): 'Regular Season', 'Playoffs', etc.

        Returns:
            pd.DataFrame: DataFrame with game information.
        """
        team_id = self._get_nba_team_id(team_abbr)

        if team_id is None:
            logger.warning("Could not find NBA team ID for: %s", team_abbr)
            return pd.DataFrame()

        try:
            self._check_and_reset_session()

            gamefinder = leaguegamefinder.LeagueGameFinder(
                team_id_nullable=team_id,
                season_nullable=season,
                season_type_nullable=season_type
            )
            time.sleep(self.base_delay)

            games = gamefinder.get_data_frames()[0]
            logger.info("Team %s: found %d games", team_abbr, len(games))

            return games

        except Exception as e:
            if self._is_rate_limit_error(e):
                logger.warning(
                    "Rate limit/connection issue for %s - resetting session and retrying...",
                    team_abbr
                )
                self._reset_session()
                time.sleep(self.batch_cooldown)
                return self._get_team_games(team_abbr, season, season_type)
            raise

    def _get_game_boxscore(
        self,
        game_id: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Get player and team boxscore for a specific game.

        Args:
            game_id (str): NBA game ID.

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
                (player_stats, team_stats) or (None, None) on failure.
        """
        if game_id in self._fetched_game_ids:
            return None, None

        try:
            self._check_and_reset_session()

            boxscore = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
            time.sleep(self.base_delay)

            player_stats = boxscore.get_data_frames()[0]
            team_stats = boxscore.get_data_frames()[1]

            self._fetched_game_ids.add(game_id)

            # Remove from failed games if previously failed
            if game_id in self._failed_games:
                del self._failed_games[game_id]

            return player_stats, team_stats

        except Exception as e:
            error_msg = str(e)

            if self._is_rate_limit_error(e):
                logger.warning(
                    "Rate limit/connection issue for game %s - resetting session...",
                    game_id
                )
                self._reset_session()
                time.sleep(self.batch_cooldown)
                return self._retry_single_game_boxscore(game_id)

            logger.error("Error fetching boxscore for game %s: %s", game_id, error_msg)
            self._record_failed_game(game_id, error_msg)
            return None, None

    def _retry_single_game_boxscore(
        self,
        game_id: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Single retry attempt for a game after session reset.

        Args:
            game_id (str): NBA game ID.

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
                (player_stats, team_stats) or (None, None) on failure.
        """
        try:
            self._request_count += 1

            boxscore = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
            time.sleep(self.base_delay)

            player_stats = boxscore.get_data_frames()[0]
            team_stats = boxscore.get_data_frames()[1]

            self._fetched_game_ids.add(game_id)

            if game_id in self._failed_games:
                del self._failed_games[game_id]

            logger.info("Game %s succeeded after retry", game_id)
            return player_stats, team_stats

        except Exception as e:
            error_msg = str(e)
            logger.error("Retry failed for game %s: %s", game_id, error_msg)
            self._record_failed_game(game_id, error_msg)
            return None, None

    # -------------------------------------------------------------------------
    # Private Methods - Session Management
    # -------------------------------------------------------------------------

    def _reset_session(self) -> None:
        """
        Reset the HTTP session by reloading nba_api modules.

        The nba_api library manages its own internal HTTP session.
        To truly reset connections, we must reload the modules themselves.
        """
        import nba_api.stats.endpoints.boxscoretraditionalv3
        import nba_api.stats.endpoints.leaguegamefinder
        import nba_api.stats.library.http

        importlib.reload(nba_api.stats.library.http)
        importlib.reload(nba_api.stats.endpoints.boxscoretraditionalv3)
        importlib.reload(nba_api.stats.endpoints.leaguegamefinder)

        gc.collect()
        self._request_count = 0

        logger.info("Session reset - nba_api modules reloaded")

    def _check_and_reset_session(self) -> None:
        """
        Check if session reset is needed based on request count.

        Triggers a session reset and cooldown when the batch size is reached.
        """
        self._request_count += 1

        if self._request_count >= self.batch_size:
            logger.info(
                "Reached %d requests - initiating batch cooldown and session reset",
                self._request_count
            )
            time.sleep(self.batch_cooldown)
            self._reset_session()

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """
        Detect if an error is related to rate limiting or connection exhaustion.

        Args:
            error (Exception): The exception to check.

        Returns:
            bool: True if rate limit or connection related, False otherwise.
        """
        error_str = str(error).lower()
        return any(indicator in error_str for indicator in RATE_LIMIT_INDICATORS)

    def _record_failed_game(self, game_id: str, failure_reason: str) -> None:
        """
        Record a failed game for later retry.

        Args:
            game_id (str): NBA game ID.
            failure_reason (str): Description of the failure.
        """
        self._failed_games[game_id] = FailedGameRecord(
            game_id=game_id,
            failure_reason=failure_reason,
            timestamp=datetime.now().isoformat()
        )
        logger.warning("Game %s added to failed cache", game_id)

    # -------------------------------------------------------------------------
    # Private Methods - Team Mapping
    # -------------------------------------------------------------------------

    def _get_nba_team_id(self, team_abbr: str) -> Optional[int]:
        """
        Get NBA API team ID from team abbreviation.

        Args:
            team_abbr (str): Team abbreviation (e.g., 'BOS').

        Returns:
            Optional[int]: NBA team ID or None if not found.
        """
        return self.team_mapping.get(team_abbr, {}).get('team_id')

    def _get_team_mapping(self) -> Dict[str, Dict]:
        """
        Get the team mapping dictionary.

        Returns:
            Dict[str, Dict]: Mapping dictionary with team IDs.

        .. deprecated::
            Use the module-level NBA_TEAM_MAPPING constant instead.
        """
        return NBA_TEAM_MAPPING

    # -------------------------------------------------------------------------
    # Private Methods - Column Mapping
    # -------------------------------------------------------------------------

    def _map_player_boxscore_columns(
        self,
        df: pd.DataFrame,
        config: Dict
    ) -> pd.DataFrame:
        """
        Map NBA API player boxscore columns to standard format.

        Args:
            df (pd.DataFrame): Raw player stats from NBA API.
            config (Dict): Configuration dictionary.

        Returns:
            pd.DataFrame: DataFrame with standardized column names.
        """
        column_mapping = {
            'gameId': 'game_id',
            'teamId': 'team_id',
            'teamTricode': 'tm',
            'teamCity': 'team_city',
            'personId': 'player_id',
            'familyName': 'player_name',
            'playerSlug': 'nickname',
            'position': 'start_position',
            'comment': 'comment',
            'minutes': 'min',
            'fieldGoalsMade': 'fg',
            'fieldGoalsAttempted': 'fga',
            'fieldGoalsPercentage': 'fg_pct',
            'threePointersMade': 'fg3',
            'threePointersAttempted': 'fg3a',
            'threePointersPercentage': 'fg3_pct',
            'freeThrowsMade': 'ft',
            'freeThrowsAttempted': 'fta',
            'freeThrowsPercentage': 'ft_pct',
            'reboundsOffensive': 'orb',
            'reboundsDefensive': 'drb',
            'reboundsTotal': 'trb',
            'assists': 'ast',
            'steals': 'stl',
            'blocks': 'blk',
            'turnovers': 'tov',
            'foulsPersonal': 'pf',
            'points': 'pts',
            'plusMinusPoints': 'plus_minus',
        }

        df = df.rename(columns=column_mapping)

        return df

    def _map_team_boxscore_columns(
        self,
        df: pd.DataFrame,
        config: Dict
    ) -> pd.DataFrame:
        """
        Map NBA API team boxscore columns to standard format.

        Args:
            df (pd.DataFrame): Raw team stats from NBA API.
            config (Dict): Configuration dictionary.

        Returns:
            pd.DataFrame: DataFrame with standardized column names.
        """
        column_mapping = {
            'GAME_ID': 'game_id',
            'TEAM_ID': 'team_id',
            'TEAM_ABBREVIATION': 'tm',
            'TEAM_CITY': 'team_city',
            'TEAM_NAME': 'team_name',
            'MIN': 'min',
            'FGM': 'fg',
            'FGA': 'fga',
            'FG_PCT': 'fg_pct',
            'FG3M': 'fg3',
            'FG3A': 'fg3a',
            'FG3_PCT': 'fg3_pct',
            'FTM': 'ft',
            'FTA': 'fta',
            'FT_PCT': 'ft_pct',
            'OREB': 'orb',
            'DREB': 'drb',
            'REB': 'trb',
            'AST': 'ast',
            'STL': 'stl',
            'BLK': 'blk',
            'TO': 'tov',
            'PF': 'pf',
            'PTS': 'pts',
            'PLUS_MINUS': 'plus_minus',
        }

        df = df.rename(columns=column_mapping)

        return df

    # -------------------------------------------------------------------------
    # Private Methods - Configuration
    # -------------------------------------------------------------------------

    def _convert_season_format(self, season: int) -> str:
        """
        Convert season year to NBA API format.

        Args:
            season (int): Season year (e.g., 2024).

        Returns:
            str: Season in NBA API format (e.g., "2023-24").
        """
        return f"{season - 1}-{str(season)[-2:]}"

    def _filter_teams(self, team_city_refdata: pd.DataFrame) -> pd.DataFrame:
        """
        Filter teams based on feature input and season validity.

        This method filters teams in two stages:
        1. Based on user input (specific team, list of teams, or 'all')
        2. Based on season validity (excludes teams not active in requested season)

        Args:
            team_city_refdata (pd.DataFrame): Full team reference data.

        Returns:
            pd.DataFrame: Filtered team data based on the team parameter
                and season validity.
        """
        team = self.feature_object.team
        season = self.feature_object.season

        # First filter: based on user input
        if isinstance(team, list):
            filtered_df = team_city_refdata[
                team_city_refdata["team_abrev"].isin(team)
            ]
        elif isinstance(team, str) and team != "all":
            filtered_df = team_city_refdata[
                team_city_refdata["team_abrev"] == team
            ]
        else:
            filtered_df = team_city_refdata

        # Second filter: based on season validity
        # This prevents duplicate fetching for teams with historical name changes
        all_teams = list(filtered_df["team_abrev"])
        valid_teams = [
            team_abbr for team_abbr in all_teams
            if is_team_valid_for_season(team_abbr, season)
        ]

        # Log any teams filtered out due to season validity
        excluded_teams = get_excluded_teams_for_season(all_teams, season)
        if excluded_teams:
            logger.info(
                "Excluded teams not valid for season %d: %s",
                season,
                excluded_teams
            )

        return filtered_df[filtered_df["team_abrev"].isin(valid_teams)]

    def _load_config(self) -> Dict:
        """
        Load configuration from params.yaml.

        Returns:
            Dict: Configuration dictionary containing column mappings
                and output specifications.
        """
        ref = (
            importlib_resources.files("basketball_reference_webscrapper")
            / "params.yaml"
        )
        with importlib_resources.as_file(ref) as path:
            with open(path, encoding="utf-8") as conf_file:
                return yaml.safe_load(conf_file)

    def _load_team_refdata(self) -> pd.DataFrame:
        """
        Load team reference data from CSV file.

        Returns:
            pd.DataFrame: DataFrame containing team abbreviations and metadata.
        """
        ref = (
            importlib_resources.files("basketball_reference_webscrapper")
            / "constants/team_city_refdata.csv"
        )
        with importlib_resources.as_file(ref) as path:
            return pd.read_csv(path, sep=";")

    def _get_season_type(self) -> str:
        """
        Get season type string for NBA API.

        Returns:
            str: Season type string ('Regular Season', 'Playoffs', etc.).
        """
        # Default to Regular Season, can be extended based on feature_object
        return "Regular Season"

    # -------------------------------------------------------------------------
    # Private Methods - Validation
    # -------------------------------------------------------------------------

    def _validate_data_type(self) -> None:
        """
        Validate data_type input parameter.

        Raises:
            ValueError: If data_type is None or not in the list of
                supported data types.
        """
        if self.feature_object.data_type is None:
            raise ValueError(
                "data_type is a required argument. "
                f"Accepted values are: {VALID_DATA_TYPES}. "
                "Read documentation for more details."
            )

        if str(self.feature_object.data_type) not in VALID_DATA_TYPES:
            raise ValueError(
                f"data_type value '{self.feature_object.data_type}' is not supported. "
                f"Accepted values are: {VALID_DATA_TYPES}. "
                "Read documentation for more details."
            )

    def _validate_season(self) -> None:
        """
        Validate season input parameter.

        Raises:
            ValueError: If season is not an integer or is before the
                minimum supported season (2000).
        """
        if not isinstance(self.feature_object.season, int):
            raise ValueError(
                "season is a required argument and should be an int value "
                f"between {MIN_SUPPORTED_SEASON} and current NBA season."
            )

        if self.feature_object.season < MIN_SUPPORTED_SEASON:
            raise ValueError(
                f"season value {self.feature_object.season} is not supported. "
                f"It should be between {MIN_SUPPORTED_SEASON} and current NBA season."
            )

    def _validate_team(self, team_abbrev_list: List[str]) -> None:
        """
        Validate team input parameter.

        Args:
            team_abbrev_list (List[str]): List of valid team abbreviations.

        Raises:
            ValueError: If team is not 'all', a valid abbreviation, or a
                list of valid abbreviations.
        """
        team = self.feature_object.team

        if isinstance(team, list):
            if not all(isinstance(t, str) for t in team):
                raise ValueError(
                    "team list must contain only string values. "
                    "Each value should be a valid NBA team abbreviation."
                )
            if not set(team).issubset(set(team_abbrev_list)):
                invalid_teams = set(team) - set(team_abbrev_list)
                raise ValueError(
                    f"Invalid team abbreviation(s): {invalid_teams}. "
                    "Value needs to be 'all' or a valid NBA team abbreviation "
                    "such as 'BOS' for Boston Celtics."
                )
        elif isinstance(team, str):
            if team != "all" and team not in team_abbrev_list:
                raise ValueError(
                    f"team value '{team}' is not accepted. "
                    "Value needs to be 'all' or a valid NBA team abbreviation "
                    "such as 'BOS' for Boston Celtics."
                )
        else:
            raise ValueError(
                f"team must be a string or list of strings, got {type(team).__name__}. "
                "Value needs to be 'all' or a valid NBA team abbreviation "
                "such as 'BOS' for Boston Celtics."
            )

    # -------------------------------------------------------------------------
    # Deprecated Methods - Backward Compatibility
    # -------------------------------------------------------------------------

    def _get_data_type_validation(self) -> None:
        """
        Validate data_type input (deprecated).

        .. deprecated::
            Use :meth:`_validate_data_type` instead.
        """
        self._validate_data_type()

    def _get_season_validation(self) -> None:
        """
        Validate season input (deprecated).

        .. deprecated::
            Use :meth:`_validate_season` instead.
        """
        self._validate_season()

    def _get_team_list_values_validation(self, team_abbrev_list: List[str]) -> None:
        """
        Validate team input (deprecated).

        .. deprecated::
            Use :meth:`_validate_team` instead.

        Args:
            team_abbrev_list (List[str]): List of valid team abbreviations.
        """
        self._validate_team(team_abbrev_list)