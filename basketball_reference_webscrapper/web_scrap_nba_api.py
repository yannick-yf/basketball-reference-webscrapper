"""
Class that fetches NBA data from the official NBA API (stats.nba.com)
using the nba_api Python package.

This module provides an alternative data source to Basketball Reference
by using the NBA's official stats API through the nba_api Python package.
"""

# Standard library
import time
from typing import Dict, List, Optional, Union

# Third-party
import pandas as pd
import requests
import importlib_resources
import yaml
from nba_api.stats.endpoints.teamgamelogs import TeamGameLogs
from nba_api.stats.endpoints.leaguegamefinder import LeagueGameFinder
from nba_api.stats.library.parameters import SeasonType

# Local
from basketball_reference_webscrapper.data_models.feature_model import FeatureIn
from basketball_reference_webscrapper.utils.logs import get_logger

__all__ = ['WebScrapNBAApi']

logger = get_logger("SCRAPER_NBA_API", log_level="INFO")

# Constants
VALID_DATA_TYPES = ["gamelog", "schedule", "schedule_non_played_games"]
MIN_SUPPORTED_SEASON = 2000
DEFAULT_REQUEST_DELAY = 0.4  # seconds between requests

# NBA Team ID mapping (current as of 2024-25 season)
NBA_TEAM_MAPPING: Dict[str, Dict] = {
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


class WebScrapNBAApi:
    """
    Class that fetches NBA data from the official NBA API using the nba_api package.

    This class provides an alternative data source to Basketball Reference
    by using the NBA's official stats API through the nba_api Python package.
    It maintains the same output structure and column names for compatibility.

    Note:
        NBA API blocks cloud providers (AWS, Heroku, GCP, etc.).
        Use this class for local development only.

    Attributes:
        feature_object (FeatureIn): Input feature object containing data_type,
            season, and team parameters.
        team_mapping (Dict): Mapping between team abbreviations and NBA API team IDs.

    Example:
        >>> from basketball_reference_webscrapper.data_models.feature_model import FeatureIn
        >>> feature = FeatureIn(data_type='gamelog', season=2023, team='BOS')
        >>> scraper = WebScrapNBAApi(feature_object=feature)
        >>> data = scraper.fetch_nba_api_data()
        >>> print(data.head())
    """

    def __init__(
        self,
        feature_object: FeatureIn,
        request_delay: float = DEFAULT_REQUEST_DELAY
    ) -> None:
        """
        Initialize the NBA API scraper.

        Args:
            feature_object (FeatureIn): Feature object containing data_type,
                season, and team parameters.
            request_delay (float): Delay in seconds between API requests to
                respect rate limiting. Defaults to 0.4 seconds (~150 req/min).
        """
        self.feature_object = feature_object
        self.request_delay = request_delay
        self.team_mapping = NBA_TEAM_MAPPING

    def fetch_nba_api_data(self) -> pd.DataFrame:
        """
        Fetch NBA data from the official NBA API.

        This is the main entry point for fetching data. It validates inputs,
        iterates through teams, and fetches the appropriate data based on
        the configured data_type.

        Returns:
            pd.DataFrame: DataFrame with columns matching Basketball Reference
                format where applicable. Returns empty DataFrame with expected
                columns if no data is found.

        Raises:
            ValueError: If data_type, season, or team parameters are invalid.

        Example:
            >>> feature = FeatureIn(data_type='gamelog', season=2023, team='BOS')
            >>> scraper = WebScrapNBAApi(feature_object=feature)
            >>> data = scraper.fetch_nba_api_data()
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
                elif self.feature_object.data_type == "schedule_non_played_games":
                    team_data = self._fetch_schedule_data_non_played_games()
                    nba_api_data = team_data
                    break
                else:
                    logger.warning("Unsupported data type: %s", self.feature_object.data_type)
                    continue

                if team_data is not None and not team_data.empty:
                    team_data["id_season"] = self.feature_object.season
                    team_data["tm"] = team_abrev
                    nba_api_data = pd.concat(
                        [nba_api_data, team_data],
                        axis=0,
                        ignore_index=True
                    )
                    logger.info(
                        "Successfully fetched %d records for team: %s",
                        len(team_data),
                        team_abrev
                    )
                else:
                    logger.warning("No data found for team: %s", team_abrev)

            except Exception as e:
                logger.error("Error fetching data for team %s: %s", team_abrev, str(e))

            # Rate limiting to respect NBA API limits
            time.sleep(self.request_delay)

        # Return empty DataFrame with correct columns if no data
        if nba_api_data.empty:
            logger.warning(
                "No data was fetched for any team. "
                "Returning empty DataFrame with expected columns."
            )
            config_key = f"{self.feature_object.data_type}_nba_api"
            return pd.DataFrame(columns=config[config_key]["list_columns_to_select"])

        # Select and return only the columns that exist in the data
        config_key = f"{self.feature_object.data_type}_nba_api"
        final_columns = config[config_key]["list_columns_to_select"]
        existing_columns = [col for col in final_columns if col in nba_api_data.columns]

        if len(existing_columns) < len(final_columns):
            missing_cols = set(final_columns) - set(existing_columns)
            logger.info(
                "Note: %d columns not available from API: %s",
                len(missing_cols),
                missing_cols
            )

        nba_api_data = nba_api_data[existing_columns]

        logger.info(
            "Successfully fetched data: %d total records",
            len(nba_api_data)
        )

        return nba_api_data

    # -------------------------------------------------------------------------
    # Private Methods - Data Fetching
    # -------------------------------------------------------------------------

    def _fetch_gamelog_data(self, team_abrev: str, config: Dict) -> Optional[pd.DataFrame]:
        """
        Fetch game log data for a specific team from NBA API.

        Args:
            team_abrev (str): Team abbreviation (e.g., 'BOS').
            config (Dict): Configuration dictionary.

        Returns:
            Optional[pd.DataFrame]: Game log data with mapped column names,
                or None if no data found.

        Note:
            Opponent statistics are not included in NBA API gamelog endpoint.
            These would require separate API calls per game.
        """
        # Convert season to NBA API format (e.g., 2024 -> "2023-24")
        season_str = self._convert_season_format(self.feature_object.season)

        # Get NBA team ID from abbreviation
        nba_team_id = self._get_nba_team_id(team_abrev)

        if nba_team_id is None:
            logger.warning("Could not find NBA team ID for: %s", team_abrev)
            return None

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
                return None

            # Map NBA API columns to Basketball Reference format
            df = self._map_gamelog_columns(df, team_abrev)

            return df

        except Exception as e:
            logger.error(
                "Error fetching gamelog data for team %s: %s",
                team_abrev,
                str(e)
            )
            return None

    def _fetch_schedule_data(self, team_abrev: str, config: Dict) -> Optional[pd.DataFrame]:
        """
        Fetch schedule data for a specific team from NBA API.

        Args:
            team_abrev (str): Team abbreviation (e.g., 'BOS').
            config (Dict): Configuration dictionary.

        Returns:
            Optional[pd.DataFrame]: Schedule data with mapped column names,
                or None if no data found.
        """
        # Convert season to NBA API format
        season_str = self._convert_season_format(self.feature_object.season)

        # Get NBA team ID
        nba_team_id = self._get_nba_team_id(team_abrev)

        if nba_team_id is None:
            logger.warning("Could not find NBA team ID for: %s", team_abrev)
            return None

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
                return None

            # Map NBA API columns to Basketball Reference format
            df = self._map_schedule_columns(df, team_abrev)

            return df

        except Exception as e:
            logger.error(
                "Error fetching schedule data for team %s: %s",
                team_abrev,
                str(e)
            )
            return None

    def _fetch_schedule_data_non_played_games(self) -> pd.DataFrame:
        """
        Fetch full season schedule including non-played games from NBA API.

        This method fetches the complete season schedule directly from the
        NBA's data endpoint, which includes future games that haven't been
        played yet.

        Returns:
            pd.DataFrame: Schedule data with game_id, game_date, tm, and opp columns.
        """
        url = (
            f"http://data.nba.com/data/10s/v2015/json/mobile_teams/nba/"
            f"{self.feature_object.season - 1}/league/00_full_schedule.json"
        )

        response = requests.get(url, timeout=60)
        schedule_data = response.json()

        # Process the data to extract games
        all_games = []
        for month in schedule_data['lscd']:
            for game in month['mscd']['g']:
                game_info = {
                    'game_id': game['gid'],
                    'game_date': game['gdte'],
                    'tm': game['h']['ta'],
                    'opp': game['v']['ta']
                }
                all_games.append(game_info)

        schedule_df = pd.DataFrame(all_games)

        logger.info(
            "Fetched %d games from full schedule",
            len(schedule_df)
        )

        return schedule_df

    # -------------------------------------------------------------------------
    # Private Methods - Column Mapping
    # -------------------------------------------------------------------------

    def _map_gamelog_columns(self, df: pd.DataFrame, team_abrev: str) -> pd.DataFrame:
        """
        Map NBA API gamelog columns to Basketball Reference format.

        Args:
            df (pd.DataFrame): Raw data from NBA API.
            team_abrev (str): Team abbreviation.

        Returns:
            pd.DataFrame: DataFrame with Basketball Reference column names.
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
            df['extdom'] = df['matchup_raw'].apply(
                lambda x: '@' if ' @ ' in str(x) else ''
            )
            df['opp'] = df['matchup_raw'].apply(
                lambda x: str(x).split(' ')[-1] if pd.notna(x) else ''
            )
        else:
            df['extdom'] = ''
            df['opp'] = ''

        # Convert percentage columns to string format
        pct_columns = ['fg_prct_tm', '3p_prct_tm', 'ft_prct_tm']
        for col in pct_columns:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else None
                )

        return df

    def _map_schedule_columns(self, df: pd.DataFrame, team_abrev: str) -> pd.DataFrame:
        """
        Map NBA API schedule columns to Basketball Reference format.

        Args:
            df (pd.DataFrame): Raw data from NBA API.
            team_abrev (str): Team abbreviation.

        Returns:
            pd.DataFrame: DataFrame with Basketball Reference column names.
        """
        # NBA API columns to Basketball Reference mapping
        column_mapping = {
            'GAME_DATE': 'game_date',
            'MATCHUP': 'matchup_raw',
            'WL': 'w_l',
            'PTS': 'pts_tm',
            'MIN': 'total_minutes'
        }

        # Rename columns
        df = df.rename(columns=column_mapping)

        # Sort by game date to get chronological order
        df = df.sort_values('game_date')

        # Parse matchup to get opponent and home/away
        if 'matchup_raw' in df.columns:
            df['extdom'] = df['matchup_raw'].apply(
                lambda x: '@' if ' @ ' in str(x) else ''
            )
            df['opponent'] = df['matchup_raw'].apply(
                lambda x: str(x).split(' ')[-1] if pd.notna(x) else ''
            )
        else:
            df['extdom'] = ''
            df['opponent'] = ''

        # Add time_start (not available in API - set to empty)
        df['time_start'] = ''

        # Calculate overtime based on total minutes played
        if 'total_minutes' in df.columns:
            logger.info(
                "Calculating overtime values from minutes played for team: %s",
                team_abrev
            )
            df['overtime'] = df['total_minutes'].apply(self._get_overtime_from_minutes)
            logger.info(
                "Successfully calculated overtime data for team: %s",
                team_abrev
            )
        else:
            logger.warning("MIN column not found, setting overtime to empty")
            df['overtime'] = ''

        # Calculate running win/loss totals
        df['w_tot'] = (df['w_l'] == 'W').cumsum()
        df['l_tot'] = (df['w_l'] == 'L').cumsum()

        # Calculate win/loss streak
        df['streak_w_l'] = self._calculate_streak(df['w_l'])

        return df

    def _get_overtime_from_minutes(self, total_minutes: float) -> str:
        """
        Determine overtime periods based on total minutes played.

        Regular game: ~240 minutes (4 quarters × 12 min × 5 players)
        Each OT adds: ~25 minutes (5 min × 5 players)

        Args:
            total_minutes (float): Total minutes from MIN column.

        Returns:
            str: Empty string for regulation, 'OT' for 1 OT, '2OT', '3OT', etc.
        """
        if pd.isna(total_minutes):
            return ''

        # Calculate overtime periods
        # Regulation is ~240 minutes, each OT adds ~25 minutes
        # Use 252.5 as threshold (midpoint between 240 and 265)
        if total_minutes < 252.5:
            return ''

        # Calculate number of OT periods
        ot_periods = round((total_minutes - 240) / 25)

        if ot_periods <= 0:
            return ''
        elif ot_periods == 1:
            return 'OT'
        else:
            return f'{ot_periods}OT'

    def _calculate_streak(self, wl_series: pd.Series) -> pd.Series:
        """
        Calculate win/loss streak from W/L series.

        Args:
            wl_series (pd.Series): Series of 'W' and 'L' values.

        Returns:
            pd.Series: Streak values (e.g., 'W 3', 'L 2').
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

    # -------------------------------------------------------------------------
    # Private Methods - Team Mapping
    # -------------------------------------------------------------------------

    def _get_nba_team_id(self, team_abrev: str) -> Optional[int]:
        """
        Get NBA API team ID from team abbreviation.

        Args:
            team_abrev (str): Team abbreviation (e.g., 'BOS').

        Returns:
            Optional[int]: NBA team ID or None if not found.
        """
        return self.team_mapping.get(team_abrev, {}).get('team_id')

    def _get_team_mapping(self) -> Dict[str, Dict]:
        """
        Get the team mapping dictionary.

        Returns:
            Dict[str, Dict]: Mapping dictionary with team IDs and NBA abbreviations.

        .. deprecated::
            Use the module-level NBA_TEAM_MAPPING constant instead.
        """
        return NBA_TEAM_MAPPING

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
        Filter teams based on feature input.

        Args:
            team_city_refdata (pd.DataFrame): Full team reference data.

        Returns:
            pd.DataFrame: Filtered team data based on the team parameter.
        """
        team = self.feature_object.team

        if isinstance(team, list):
            return team_city_refdata[
                team_city_refdata["team_abrev"].isin(team)
            ]
        elif isinstance(team, str) and team != "all":
            return team_city_refdata[
                team_city_refdata["team_abrev"] == team
            ]

        return team_city_refdata

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
                f"data_type value '{self.feature_object.data_type}' is not supported "
                "by the NBA API scraper. "
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