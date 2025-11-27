"""
NBA Boxscore Extractor - Production Ready
Extract player boxscore data for NBA games using nba_api.
Iterates team-by-team to avoid G-League data contamination.
"""

from dataclasses import dataclass, field
from datetime import datetime
import gc
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv2, leaguegamefinder
from nba_api.stats.static import teams


# Logger setup
def get_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Create and configure a logger instance.

    Args:
        name (str): Logger name
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level.upper()))
    return logger


logger = get_logger("NBA_BOXSCORE_EXTRACTOR", log_level="INFO")


# NBA Teams only (excludes G-League)
NBA_TEAMS: List[Dict] = teams.get_teams()


@dataclass
class FailedGameRecord:
    """
    Data class representing a failed game fetch attempt.

    Attributes:
        game_id (str): NBA game ID
        failure_reason (str): Description of the failure
        timestamp (str): ISO format timestamp of the failure
    """
    game_id: str
    failure_reason: str
    timestamp: str


@dataclass
class NBABoxscoreExtractor:
    """
    Extract NBA boxscore data with production-ready error handling.

    This class provides robust NBA boxscore extraction with:
    - Team-by-team iteration to ensure only NBA games (no G-League)
    - Game ID deduplication to avoid redundant API calls
    - Periodic session reset to avoid connection exhaustion
    - Caching system for failed games with retry capability

    Attributes:
        base_delay (float): Base delay between API calls in seconds
        rate_limit_wait (float): Wait time in seconds when rate limited (default 5 minutes)
        batch_size (int): Number of requests before forcing a session reset
        batch_cooldown (float): Cooldown time in seconds between batches
        cache_dir (Path): Directory for storing failed games cache
    """

    base_delay: float = 0.6
    rate_limit_wait: float = 300.0  # 5 minutes
    batch_size: int = 500  # Reset session every 500 requests
    batch_cooldown: float = 60.0  # 1 minute cooldown between batches
    cache_dir: Path = field(default_factory=lambda: Path("./cache"))

    def __post_init__(self) -> None:
        """Initialize extractor with state tracking."""
        self._fetched_game_ids: Set[str] = set()
        self._failed_games: Dict[str, FailedGameRecord] = {}
        self._request_count: int = 0

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "NBABoxscoreExtractor initialized with base_delay=%.2fs, batch_size=%d",
            self.base_delay, self.batch_size
        )

    def _reset_session(self) -> None:
        """
        Reset the HTTP session by reloading nba_api modules.

        The nba_api library manages its own internal HTTP session.
        To truly reset connections, we must reload the modules themselves.
        This forces Python to re-initialize all module-level state including
        any cached sessions or connection pools.
        """
        import importlib
        import nba_api.stats.endpoints.boxscoretraditionalv2
        import nba_api.stats.endpoints.leaguegamefinder
        import nba_api.stats.library.http

        # Reload the http library first (handles actual requests)
        importlib.reload(nba_api.stats.library.http)

        # Reload the endpoint modules
        importlib.reload(nba_api.stats.endpoints.boxscoretraditionalv2)
        importlib.reload(nba_api.stats.endpoints.leaguegamefinder)

        # Force garbage collection to clean up old connections
        gc.collect()

        # Reset request counter
        self._request_count = 0

        logger.info("Session reset - nba_api modules reloaded, fresh connections established")

    def _check_and_reset_session(self) -> None:
        """
        Check if session reset is needed based on request count.

        Called before each API request to ensure connection health.
        """
        self._request_count += 1

        if self._request_count >= self.batch_size:
            logger.info(
                "Reached %d requests - initiating batch cooldown and session reset",
                self._request_count
            )
            time.sleep(self.batch_cooldown)
            self._reset_session()

    def get_nba_team_abbreviations(self) -> List[str]:
        """
        Get list of all NBA team abbreviations (excludes G-League).

        Returns:
            List[str]: List of 30 NBA team abbreviations
        """
        return [team["abbreviation"] for team in NBA_TEAMS]

    def get_team_games(
        self,
        team_abbr: str,
        season: str = "2023-24",
        season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        """
        Get all games for a specific team.

        Args:
            team_abbr (str): Team abbreviation (e.g., 'LAL', 'BOS')
            season (str): Season in format 'YYYY-YY' (e.g., '2023-24')
            season_type (str): 'Regular Season', 'Playoffs', 'All Star'

        Returns:
            pd.DataFrame: DataFrame with game information

        Raises:
            ValueError: If team abbreviation is invalid
        """
        team_id = self._get_team_id_from_abbr(team_abbr)
        if team_id is None:
            raise ValueError(f"Invalid team abbreviation: {team_abbr}")

        try:
            # Check if session reset is needed
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
                return self.get_team_games(team_abbr, season, season_type)
            raise

    def get_all_season_games(
        self,
        season: str = "2023-24",
        season_type: str = "Regular Season",
        team_abbr: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get all games for the season, iterating team-by-team to ensure NBA-only data.

        Args:
            season (str): Season in format 'YYYY-YY' (e.g., '2023-24')
            season_type (str): 'Regular Season', 'Playoffs', 'All Star'
            team_abbr (Optional[str]): Specific team, or None for all NBA teams

        Returns:
            pd.DataFrame: DataFrame with all unique games
        """
        if team_abbr:
            # Single team requested
            return self.get_team_games(team_abbr, season, season_type)

        # All NBA teams - iterate one by one
        logger.info("Fetching games for all 30 NBA teams, season %s", season)
        team_abbreviations = self.get_nba_team_abbreviations()

        all_games: List[pd.DataFrame] = []
        seen_game_ids: Set[str] = set()

        for idx, abbr in enumerate(team_abbreviations, 1):
            logger.info("Fetching games for team %d/30: %s", idx, abbr)

            try:
                team_games = self.get_team_games(abbr, season, season_type)

                if team_games.empty:
                    continue

                # Filter out already seen games
                new_games = team_games[~team_games["GAME_ID"].isin(seen_game_ids)]

                if not new_games.empty:
                    all_games.append(new_games)
                    seen_game_ids.update(new_games["GAME_ID"].tolist())
                    logger.info(
                        "Team %s: %d new games added (total unique: %d)",
                        abbr, len(new_games), len(seen_game_ids)
                    )

            except Exception as e:
                logger.error("Error fetching games for team %s: %s", abbr, str(e))

        if all_games:
            combined = pd.concat(all_games, ignore_index=True)
            logger.info("Total unique games found: %d", len(combined))
            return combined

        return pd.DataFrame()

    def get_game_boxscore(
        self,
        game_id: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Get player boxscore for a specific game.

        Args:
            game_id (str): NBA game ID

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
                (player_stats_df, team_stats_df) or (None, None) on failure
        """
        # Skip if already fetched
        if game_id in self._fetched_game_ids:
            logger.debug("Game %s already fetched, skipping", game_id)
            return None, None

        try:
            # Check if session reset is needed
            self._check_and_reset_session()

            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            time.sleep(self.base_delay)

            player_stats = boxscore.get_data_frames()[0]
            team_stats = boxscore.get_data_frames()[1]

            # Mark as fetched
            self._fetched_game_ids.add(game_id)

            # Remove from failed games if previously failed
            if game_id in self._failed_games:
                del self._failed_games[game_id]
                logger.info("Game %s succeeded, removed from failed cache", game_id)

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
                # Retry once after session reset
                return self._retry_game_boxscore(game_id)

            # Non-rate-limit error - record as failed
            logger.error("Error fetching boxscore for game %s: %s", game_id, error_msg)
            self._record_failed_game(game_id, error_msg)
            return None, None

    def _retry_game_boxscore(
        self,
        game_id: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Single retry attempt for a game after session reset.

        Args:
            game_id (str): NBA game ID

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
                (player_stats_df, team_stats_df) or (None, None) on failure
        """
        try:
            # Increment request count (session was just reset, so this starts fresh)
            self._request_count += 1

            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
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

    def extract_all_boxscores(
        self,
        season: str = "2023-24",
        season_type: str = "Regular Season",
        team_abbr: Optional[str] = None,
        max_games: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract boxscores for all games, iterating team-by-team.

        This method ensures only NBA games are fetched (no G-League) by:
        1. Getting games for each of the 30 NBA teams
        2. Tracking already-fetched game IDs to avoid duplicates

        Args:
            season (str): Season in format 'YYYY-YY' (e.g., '2023-24')
            season_type (str): 'Regular Season', 'Playoffs', 'All Star'
            team_abbr (Optional[str]): Specific team, or None for all NBA teams
            max_games (Optional[int]): Maximum number of games to process (None for all)

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (all_player_stats_df, all_team_stats_df)
        """
        # Reset state for fresh extraction
        self._fetched_game_ids.clear()

        if team_abbr:
            # Single team mode
            teams_to_process = [team_abbr]
        else:
            # All NBA teams
            teams_to_process = self.get_nba_team_abbreviations()

        logger.info(
            "Extracting boxscores for %d team(s), season %s",
            len(teams_to_process), season
        )

        all_player_stats: List[pd.DataFrame] = []
        all_team_stats: List[pd.DataFrame] = []
        games_processed = 0

        for team_idx, abbr in enumerate(teams_to_process, 1):
            logger.info(
                "Processing team %d/%d: %s",
                team_idx, len(teams_to_process), abbr
            )

            # Get games for this team
            try:
                team_games = self.get_team_games(abbr, season, season_type)
            except Exception as e:
                logger.error("Failed to get games for team %s: %s", abbr, str(e))
                continue

            if team_games.empty:
                logger.warning("No games found for team %s", abbr)
                continue

            # Process each game
            game_ids = team_games["GAME_ID"].unique()

            for game_id in game_ids:
                # Check max_games limit
                if max_games and games_processed >= max_games:
                    logger.info("Reached max_games limit (%d)", max_games)
                    break

                # Skip already fetched
                if game_id in self._fetched_game_ids:
                    continue

                player_stats, team_stats = self.get_game_boxscore(game_id)

                if player_stats is not None and team_stats is not None:
                    all_player_stats.append(player_stats)
                    all_team_stats.append(team_stats)
                    games_processed += 1

                    if games_processed % 10 == 0:
                        logger.info("Progress: %d games processed", games_processed)

            # Check max_games limit after team
            if max_games and games_processed >= max_games:
                break

        success_count = len(all_player_stats)
        fail_count = len(self._failed_games)

        logger.info(
            "Extraction complete: %d succeeded, %d failed",
            success_count, fail_count
        )

        if all_player_stats:
            combined_player_stats = pd.concat(all_player_stats, ignore_index=True)
            combined_team_stats = pd.concat(all_team_stats, ignore_index=True)
            return combined_player_stats, combined_team_stats

        return pd.DataFrame(), pd.DataFrame()

    def retry_failed_games(
        self,
        cache_file: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retry fetching boxscores for previously failed games.

        Args:
            cache_file (Optional[str]): Path to failed games cache file.
                If None, uses in-memory failed games.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (player_stats_df, team_stats_df) for recovered games
        """
        if cache_file:
            self.load_failed_games_cache(cache_file)

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

            player_stats, team_stats = self.get_game_boxscore(game_id)

            if player_stats is not None and team_stats is not None:
                all_player_stats.append(player_stats)
                all_team_stats.append(team_stats)
                logger.info("Successfully recovered game %s", game_id)

        recovered_count = len(all_player_stats)
        logger.info(
            "Retry complete: %d/%d games recovered",
            recovered_count, len(failed_game_ids)
        )

        if all_player_stats:
            return (
                pd.concat(all_player_stats, ignore_index=True),
                pd.concat(all_team_stats, ignore_index=True)
            )

        return pd.DataFrame(), pd.DataFrame()

    def save_failed_games_cache(self, filename: Optional[str] = None) -> str:
        """
        Save failed games to a JSON cache file.

        Args:
            filename (Optional[str]): Output filename. If None, generates timestamped name.

        Returns:
            str: Path to the saved cache file
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
            filepath (str): Path to the cache file

        Returns:
            int: Number of failed games loaded
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

    def get_failed_games_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of all failed games.

        Returns:
            pd.DataFrame: Summary of failed games
        """
        if not self._failed_games:
            return pd.DataFrame(columns=["game_id", "failure_reason", "timestamp"])

        return pd.DataFrame([
            {
                "game_id": r.game_id,
                "failure_reason": r.failure_reason,
                "timestamp": r.timestamp
            }
            for r in self._failed_games.values()
        ])

    def get_fetched_game_ids(self) -> Set[str]:
        """
        Get set of already fetched game IDs.

        Returns:
            Set[str]: Set of game IDs that have been successfully fetched
        """
        return self._fetched_game_ids.copy()

    def get_request_count(self) -> int:
        """
        Get current request count since last session reset.

        Returns:
            int: Number of requests made in current session
        """
        return self._request_count

    def save_to_csv(
        self,
        player_stats_df: pd.DataFrame,
        team_stats_df: pd.DataFrame,
        prefix: str = "nba_boxscore"
    ) -> Tuple[str, str]:
        """
        Save dataframes to CSV files.

        Args:
            player_stats_df (pd.DataFrame): Player statistics
            team_stats_df (pd.DataFrame): Team statistics
            prefix (str): Prefix for output files

        Returns:
            Tuple[str, str]: (player_file_path, team_file_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        player_file = f"{prefix}_players_{timestamp}.csv"
        team_file = f"{prefix}_teams_{timestamp}.csv"

        player_stats_df.to_csv(player_file, index=False)
        team_stats_df.to_csv(team_file, index=False)

        logger.info("Data saved:")
        logger.info("  - Player stats: %s (%d rows)", player_file, len(player_stats_df))
        logger.info("  - Team stats: %s (%d rows)", team_file, len(team_stats_df))

        return player_file, team_file

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _get_team_id_from_abbr(self, team_abbr: str) -> Optional[int]:
        """
        Get team ID from abbreviation.

        Args:
            team_abbr (str): Team abbreviation

        Returns:
            Optional[int]: Team ID or None if not found
        """
        matching_teams = [
            t for t in NBA_TEAMS
            if t["abbreviation"] == team_abbr
        ]
        return matching_teams[0]["id"] if matching_teams else None

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """
        Detect if an error is related to rate limiting or connection exhaustion.

        Args:
            error (Exception): The exception to check

        Returns:
            bool: True if rate limit or connection related, False otherwise
        """
        error_str = str(error).lower()
        # Include connection-related errors that indicate session exhaustion
        rate_limit_indicators = [
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
        return any(indicator in error_str for indicator in rate_limit_indicators)

    def _record_failed_game(self, game_id: str, failure_reason: str) -> None:
        """
        Record a failed game for later retry.

        Args:
            game_id (str): NBA game ID
            failure_reason (str): Description of the failure
        """
        self._failed_games[game_id] = FailedGameRecord(
            game_id=game_id,
            failure_reason=failure_reason,
            timestamp=datetime.now().isoformat()
        )
        logger.warning("Game %s added to failed cache", game_id)


def main() -> None:
    """Example usage demonstrating all features."""

    extractor = NBABoxscoreExtractor(
        base_delay=0.6,
        rate_limit_wait=300.0,  # 5 minutes (fallback)
        batch_size=500,  # Reset session every 500 requests
        batch_cooldown=60.0,  # 1 minute cooldown between batches
        cache_dir=Path("./nba_cache")
    )

    print("\n" + "=" * 60)
    print("Available NBA Teams (30 teams, no G-League)")
    print("=" * 60)
    print(extractor.get_nba_team_abbreviations())

    print("\n" + "=" * 60)
    print("Example 1: Extract boxscores for full season (limited to 10 games)")
    print("=" * 60)

    try:
        player_stats, team_stats = extractor.extract_all_boxscores(
            season="2024-25",
            season_type="Regular Season",
            team_abbr=None,  # All NBA teams
            max_games=10
        )

        if not player_stats.empty:
            print(f"\nPlayer stats shape: {player_stats.shape}")
            print(f"Unique games fetched: {len(extractor.get_fetched_game_ids())}")

            display_cols = ["GAME_ID", "TEAM_ABBREVIATION", "PLAYER_NAME", "MIN", "PTS", "REB", "AST"]
            available_cols = [c for c in display_cols if c in player_stats.columns]
            print("\nSample player stats:")
            print(player_stats[available_cols].head(10))

            extractor.save_to_csv(player_stats, team_stats)

        print("\n" + "=" * 60)
        print("Failed games handling")
        print("=" * 60)

        failed_summary = extractor.get_failed_games_summary()
        if not failed_summary.empty:
            print(f"Failed games: {len(failed_summary)}")
            print(failed_summary)

            cache_file = extractor.save_failed_games_cache()
            print(f"Failed games cached to: {cache_file}")

            print("\nRetrying failed games...")
            recovered_players, recovered_teams = extractor.retry_failed_games()
            if not recovered_players.empty:
                print(f"Recovered {len(recovered_players)} player records")
        else:
            print("No failed games - all extractions succeeded!")

    except Exception as e:
        logger.error("Error in main execution: %s", str(e))
        raise

    print("\n" + "=" * 60)
    print("Extraction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()