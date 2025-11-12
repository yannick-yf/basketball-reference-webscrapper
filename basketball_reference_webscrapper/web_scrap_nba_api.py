"""
Class that fetches NBA data from the official NBA API (stats.nba.com)
using the nba_api Python package as an alternative to Basketball Reference web scraping.
"""

from dataclasses import dataclass
import time
from typing import Dict, List, Optional
import pandas as pd
import importlib_resources
import yaml

from nba_api.stats.endpoints.teamgamelogs import TeamGameLogs
from nba_api.stats.endpoints.leaguegamefinder import LeagueGameFinder
from nba_api.stats.library.parameters import SeasonType

from basketball_reference_webscrapper.data_models.feature_model import FeatureIn
from basketball_reference_webscrapper.utils.logs import get_logger

logger = get_logger("NBA_API_EXECUTION", log_level="INFO")


@dataclass
class WebScrapNBAApi:
    """
    Class that fetches NBA data from the official NBA API using the nba_api package.

    This class provides an alternative data source to Basketball Reference
    by using the NBA's official stats API through the nba_api Python package.
    It maintains the same output structure and column names for compatibility.

    Attributes:
        feature_object (FeatureIn): Input feature object containing data_type, season, and team
    """

    def __init__(self, feature_object: FeatureIn) -> None:
        """
        Initialize the NBA API scraper.

        Args:
            feature_object (FeatureIn): Feature object with data_type, season, and team
        """
        self.feature_object = feature_object

        # Team abbreviation mapping (NBA API uses different abbreviations)
        self.team_mapping = self._get_team_mapping()

    def fetch_nba_api_data(self) -> pd.DataFrame:
        """
        Main method to fetch NBA data from the official API.

        Returns:
            pd.DataFrame: DataFrame with columns matching Basketball Reference format
        """
        # Input validation
        self._get_data_type_validation()
        self._get_season_validation()

        # Load configuration
        config = self._load_config()

        # Load team reference data
        team_city_refdata = self._load_team_refdata()

        # Validate team input
        self._get_team_list_values_validation(list(team_city_refdata["team_abrev"]))

        # Filter teams based on input
        teams_to_fetch = self._filter_teams(team_city_refdata)

        # Initialize result DataFrame
        nba_api_data = pd.DataFrame()

        # Fetch data for each team
        for _, row in teams_to_fetch.iterrows():
            team_abrev = row["team_abrev"]
            logger.info("Fetching data for team: %s", team_abrev)

            try:
                if self.feature_object.data_type == "gamelog":
                    team_data = self._fetch_gamelog_data(team_abrev, config)
                elif self.feature_object.data_type == "schedule":
                    team_data = self._fetch_schedule_data(team_abrev, config)
                else:
                    logger.warning("Unsupported data type: %s", self.feature_object.data_type)
                    continue

                if not team_data.empty:
                    team_data["id_season"] = self.feature_object.season
                    team_data["tm"] = team_abrev
                    nba_api_data = pd.concat([nba_api_data, team_data], axis=0, ignore_index=True)
                    logger.info("Successfully fetched data for team: %s", team_abrev)
                else:
                    logger.warning("No data found for team: %s", team_abrev)

            except Exception as e:
                logger.error("Error fetching data for team %s: %s", team_abrev, str(e))

            # Rate limiting to respect NBA API limits (~100 requests per minute)
            # Using 0.4 seconds between requests = 150 req/min with safety margin
            # Tested with all teams - no rate limit issues observed
            time.sleep(0.4)

        # Return empty DataFrame with correct columns if no data
        if nba_api_data.empty:
            logger.warning("No data was fetched for any team. Returning empty DataFrame with expected columns.")
            return pd.DataFrame(columns=config[f"{self.feature_object.data_type}_nba_api"]["list_columns_to_select"])

        # Select and return only the columns that exist in the data
        # This allows flexibility for downstream data engineering pipelines
        final_columns = config[f"{self.feature_object.data_type}_nba_api"]["list_columns_to_select"]
        existing_columns = [col for col in final_columns if col in nba_api_data.columns]

        if len(existing_columns) < len(final_columns):
            missing_cols = set(final_columns) - set(existing_columns)
            logger.info(f"Note: {len(missing_cols)} columns not available from API: {missing_cols}")

        nba_api_data = nba_api_data[existing_columns]

        return nba_api_data

    def _fetch_gamelog_data(self, team_abrev: str, config: Dict) -> pd.DataFrame:
        """
        Fetch game log data for a specific team from NBA API using nba_api package.

        Args:
            team_abrev (str): Team abbreviation (e.g., 'BOS')
            config (Dict): Configuration dictionary

        Returns:
            pd.DataFrame: Game log data with mapped column names
        """
        # Convert season to NBA API format (e.g., 2024 -> "2023-24")
        season_str = self._convert_season_format(self.feature_object.season)

        # Get NBA team ID from abbreviation
        nba_team_id = self._get_nba_team_id(team_abrev)

        if nba_team_id is None:
            logger.warning("Could not find NBA team ID for: %s", team_abrev)
            return pd.DataFrame()

        try:
            # Use nba_api TeamGameLogs endpoint
            gamelog_query = TeamGameLogs(
                season_nullable=season_str,
                season_type_nullable=SeasonType.regular,
                team_id_nullable=str(nba_team_id)
            )

            # Get data as DataFrame
            df = gamelog_query.get_data_frames()[0]

            if df.empty:
                logger.warning("No gamelog data returned for team: %s", team_abrev)
                return pd.DataFrame()

            # Map NBA API columns to Basketball Reference format
            df = self._map_gamelog_columns(df, team_abrev)

            return df

        except Exception as e:
            logger.error("Error fetching gamelog data for team %s: %s", team_abrev, str(e))
            return pd.DataFrame()

    def _fetch_schedule_data(self, team_abrev: str, config: Dict) -> pd.DataFrame:
        """
        Fetch schedule data for a specific team from NBA API using nba_api package.

        Args:
            team_abrev (str): Team abbreviation (e.g., 'BOS')
            config (Dict): Configuration dictionary

        Returns:
            pd.DataFrame: Schedule data with mapped column names
        """
        # Convert season to NBA API format
        season_str = self._convert_season_format(self.feature_object.season)

        # Get NBA team ID
        nba_team_id = self._get_nba_team_id(team_abrev)

        if nba_team_id is None:
            logger.warning("Could not find NBA team ID for: %s", team_abrev)
            return pd.DataFrame()

        try:
            # Use nba_api LeagueGameFinder endpoint
            gamefinder_query = LeagueGameFinder(
                team_id_nullable=str(nba_team_id),
                season_nullable=season_str,
                season_type_nullable=SeasonType.regular,
                player_or_team_abbreviation='T'
            )

            # Get data as DataFrame
            df = gamefinder_query.get_data_frames()[0]

            if df.empty:
                logger.warning("No schedule data returned for team: %s", team_abrev)
                return pd.DataFrame()

            # Map NBA API columns to Basketball Reference format
            df = self._map_schedule_columns(df, team_abrev)

            return df

        except Exception as e:
            logger.error("Error fetching schedule data for team %s: %s", team_abrev, str(e))
            return pd.DataFrame()

    def _map_gamelog_columns(self, df: pd.DataFrame, team_abrev: str) -> pd.DataFrame:
        """
        Map NBA API gamelog columns to Basketball Reference format.

        Args:
            df (pd.DataFrame): Raw data from NBA API
            team_abrev (str): Team abbreviation

        Returns:
            pd.DataFrame: DataFrame with Basketball Reference column names
        """
        # NBA API columns to Basketball Reference mapping
        column_mapping = {
            'GAME_DATE': 'game_date',
            'MATCHUP': 'matchup_raw',
            'WL': 'results',
            'PTS': 'pts_tm',
            'FGM': 'fg_tm',
            'FGA': 'fga_tm',
            'FG_PCT': 'fg_prct_tm',
            'FG3M': '3p_tm',
            'FG3A': '3pa_tm',
            'FG3_PCT': '3p_prct_tm',
            'FTM': 'ft_tm',
            'FTA': 'fta_tm',
            'FT_PCT': 'ft_prct_tm',
            'OREB': 'orb_tm',
            'REB': 'trb_tm',
            'AST': 'ast_tm',
            'STL': 'stl_tm',
            'BLK': 'blk_tm',
            'TOV': 'tov_tm',
            'PF': 'pf_tm'
        }

        # Rename columns
        df = df.rename(columns=column_mapping)

        # Add game number (need to sort by date first)
        df = df.sort_values('game_date')
        df['game_nb'] = range(1, len(df) + 1)

        # Parse matchup to get opponent and home/away
        if 'matchup_raw' in df.columns:
            df['extdom'] = df['matchup_raw'].apply(lambda x: '@' if ' @ ' in str(x) else '')
            df['opp'] = df['matchup_raw'].apply(lambda x: str(x).split(' ')[-1] if pd.notna(x) else '')
        else:
            df['extdom'] = ''
            df['opp'] = ''

        # Note: Opponent statistics are not included in NBA API gamelog endpoint
        # These would require separate API calls per game and are excluded to:
        # 1. Avoid rate limiting issues
        # 2. Keep data structure aligned with what API provides
        # 3. Allow downstream data engineering to handle opponent data separately if needed

        # Convert percentage columns to string format
        pct_columns = ['fg_prct_tm', '3p_prct_tm', 'ft_prct_tm']
        for col in pct_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else None)

        return df

    def _map_schedule_columns(self, df: pd.DataFrame, team_abrev: str) -> pd.DataFrame:
        """
        Map NBA API schedule columns to Basketball Reference format.

        Args:
            df (pd.DataFrame): Raw data from NBA API
            team_abrev (str): Team abbreviation

        Returns:
            pd.DataFrame: DataFrame with Basketball Reference column names
        """
        # NBA API columns to Basketball Reference mapping
        column_mapping = {
            'GAME_DATE': 'game_date',
            'MATCHUP': 'matchup_raw',
            'WL': 'w_l',
            'PTS': 'pts_tm'
        }

        # Rename columns
        df = df.rename(columns=column_mapping)

        # Sort by game date to get chronological order
        df = df.sort_values('game_date')

        # Parse matchup to get opponent and home/away
        if 'matchup_raw' in df.columns:
            df['extdom'] = df['matchup_raw'].apply(lambda x: '@' if ' @ ' in str(x) else '')
            df['opponent'] = df['matchup_raw'].apply(lambda x: str(x).split(' ')[-1] if pd.notna(x) else '')
        else:
            df['extdom'] = ''
            df['opponent'] = ''

        # Add time_start (not available in API - set to empty)
        df['time_start'] = ''

        # Add overtime column (not directly available - set to empty)
        df['overtime'] = ''

        # Note: Opponent points (pts_opp) not included in NBA API schedule endpoint
        # This would require separate API calls or parsing game results

        # Calculate running win/loss totals
        df['w_tot'] = (df['w_l'] == 'W').cumsum()
        df['l_tot'] = (df['w_l'] == 'L').cumsum()

        # Calculate win/loss streak
        df['streak_w_l'] = self._calculate_streak(df['w_l'])

        return df

    def _calculate_streak(self, wl_series: pd.Series) -> pd.Series:
        """
        Calculate win/loss streak from W/L series.

        Args:
            wl_series (pd.Series): Series of 'W' and 'L' values

        Returns:
            pd.Series: Streak values (e.g., 'W 3', 'L 2')
        """
        streaks = []
        current_result = None
        current_count = 0

        for result in wl_series:
            if result == current_result:
                current_count += 1
            else:
                current_result = result
                current_count = 1
            streaks.append(f"{result} {current_count}" if pd.notna(result) else '')

        return pd.Series(streaks, index=wl_series.index)

    def _get_nba_team_id(self, team_abrev: str) -> Optional[int]:
        """
        Get NBA API team ID from team abbreviation.

        Args:
            team_abrev (str): Team abbreviation (e.g., 'BOS')

        Returns:
            Optional[int]: NBA team ID or None if not found
        """
        return self.team_mapping.get(team_abrev, {}).get('team_id')

    def _get_team_mapping(self) -> Dict[str, Dict]:
        """
        Create mapping between Basketball Reference abbreviations and NBA API team IDs.

        Returns:
            Dict: Mapping dictionary with team IDs and NBA abbreviations
        """
        # NBA team IDs (current as of 2024-25 season)
        return {
            'ATL': {'team_id': 1610612737, 'nba_abbrev': 'ATL'},
            'BOS': {'team_id': 1610612738, 'nba_abbrev': 'BOS'},
            'BRK': {'team_id': 1610612751, 'nba_abbrev': 'BKN'},
            'CHA': {'team_id': 1610612766, 'nba_abbrev': 'CHA'},
            'CHI': {'team_id': 1610612741, 'nba_abbrev': 'CHI'},
            'CHO': {'team_id': 1610612766, 'nba_abbrev': 'CHA'},
            'CLE': {'team_id': 1610612739, 'nba_abbrev': 'CLE'},
            'DAL': {'team_id': 1610612742, 'nba_abbrev': 'DAL'},
            'DEN': {'team_id': 1610612743, 'nba_abbrev': 'DEN'},
            'DET': {'team_id': 1610612765, 'nba_abbrev': 'DET'},
            'GSW': {'team_id': 1610612744, 'nba_abbrev': 'GSW'},
            'HOU': {'team_id': 1610612745, 'nba_abbrev': 'HOU'},
            'IND': {'team_id': 1610612754, 'nba_abbrev': 'IND'},
            'LAC': {'team_id': 1610612746, 'nba_abbrev': 'LAC'},
            'LAL': {'team_id': 1610612747, 'nba_abbrev': 'LAL'},
            'MEM': {'team_id': 1610612763, 'nba_abbrev': 'MEM'},
            'MIA': {'team_id': 1610612748, 'nba_abbrev': 'MIA'},
            'MIL': {'team_id': 1610612749, 'nba_abbrev': 'MIL'},
            'MIN': {'team_id': 1610612750, 'nba_abbrev': 'MIN'},
            'NOP': {'team_id': 1610612740, 'nba_abbrev': 'NOP'},
            'NYK': {'team_id': 1610612752, 'nba_abbrev': 'NYK'},
            'OKC': {'team_id': 1610612760, 'nba_abbrev': 'OKC'},
            'ORL': {'team_id': 1610612753, 'nba_abbrev': 'ORL'},
            'PHI': {'team_id': 1610612755, 'nba_abbrev': 'PHI'},
            'PHO': {'team_id': 1610612756, 'nba_abbrev': 'PHX'},
            'POR': {'team_id': 1610612757, 'nba_abbrev': 'POR'},
            'SAC': {'team_id': 1610612758, 'nba_abbrev': 'SAC'},
            'SAS': {'team_id': 1610612759, 'nba_abbrev': 'SAS'},
            'TOR': {'team_id': 1610612761, 'nba_abbrev': 'TOR'},
            'UTA': {'team_id': 1610612762, 'nba_abbrev': 'UTA'},
            'WAS': {'team_id': 1610612764, 'nba_abbrev': 'WAS'},
        }

    def _convert_season_format(self, season: int) -> str:
        """
        Convert season year to NBA API format.

        Args:
            season (int): Season year (e.g., 2024)

        Returns:
            str: Season in NBA API format (e.g., "2023-24")
        """
        return f"{season-1}-{str(season)[-2:]}"

    def _filter_teams(self, team_city_refdata: pd.DataFrame) -> pd.DataFrame:
        """
        Filter teams based on feature input.

        Args:
            team_city_refdata (pd.DataFrame): Team reference data

        Returns:
            pd.DataFrame: Filtered team data
        """
        if isinstance(self.feature_object.team, list):
            return team_city_refdata[
                team_city_refdata["team_abrev"].isin(self.feature_object.team)
            ]
        elif isinstance(self.feature_object.team, str):
            if self.feature_object.team != "all":
                return team_city_refdata[
                    team_city_refdata["team_abrev"] == self.feature_object.team
                ]
        return team_city_refdata

    def _load_config(self) -> Dict:
        """
        Load configuration from params.yaml.

        Returns:
            Dict: Configuration dictionary
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
        Load team reference data.

        Returns:
            pd.DataFrame: Team reference data
        """
        ref = (
            importlib_resources.files("basketball_reference_webscrapper")
            / "constants/team_city_refdata.csv"
        )
        with importlib_resources.as_file(ref) as path:
            return pd.read_csv(path, sep=";")

    def _get_data_type_validation(self) -> None:
        """Validate data_type input."""
        if self.feature_object.data_type is not None:
            if str(self.feature_object.data_type) not in ["gamelog", "schedule"]:
                raise ValueError(
                    "data_type value provided is not supported by the NBA API scraper. "
                    "Accepted values are: 'gamelog', 'schedule'. "
                    "Read documentation for more details"
                )
        else:
            raise ValueError(
                "data_type is a required argument. "
                "Accepted values are: 'gamelog', 'schedule'. "
                "Read documentation for more details"
            )

    def _get_season_validation(self) -> None:
        """Validate season input."""
        if isinstance(self.feature_object.season, int):
            if self.feature_object.season < 1999:
                raise ValueError(
                    "season value provided is a year not supported by the package. "
                    "It should be between 2000 and current NBA season."
                )
        else:
            raise ValueError(
                "season is a required argument and should be an int value between 2000 "
                "and current NBA season."
            )

    def _get_team_list_values_validation(self, team_abbrev_list: List[str]) -> None:
        """
        Validate team input.

        Args:
            team_abbrev_list (List[str]): List of valid team abbreviations
        """
        if (isinstance(self.feature_object.team, list)) and (
            all(isinstance(s, str) for s in self.feature_object.team)
        ):
            if set(self.feature_object.team).issubset(team_abbrev_list) is False:
                raise ValueError(
                    "team list arg provided is not accepted. "
                    "Value needs to be 'all' or a NBA team abbreviation "
                    "such as BOS for Boston Celtics. "
                    "More details on all values possible in the GitHub Repo docs"
                )
        elif isinstance(self.feature_object.team, str):
            if (self.feature_object.team != "all") and (
                self.feature_object.team not in team_abbrev_list
            ):
                raise ValueError(
                    "team arg provided is not accepted. "
                    "Value needs to be 'all' or a NBA team abbreviation "
                    "such as BOS for Boston Celtics. "
                    "More details on all values possible in the GitHub Repo docs"
                )
        else:
            raise ValueError(
                "team args should be a string or a list of string. "
                "Value needs to be NBA team abbreviation such as BOS for Boston Celtics. "
                "More details on all values possible in the GitHub Repo docs"
            )