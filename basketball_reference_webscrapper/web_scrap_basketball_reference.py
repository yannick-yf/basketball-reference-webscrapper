"""
Class that fetches NBA data from Basketball Reference website via web scraping.

This module provides web scraping functionality to extract NBA game data,
schedules, and player attributes from Basketball Reference.
"""

# Standard library
import time
from typing import Dict, List, Optional, Union
from urllib.request import urlopen, Request

# Third-party
from bs4 import BeautifulSoup
import pandas as pd
import requests
import importlib_resources
import yaml

# Local
from basketball_reference_webscrapper.data_models.feature_model import FeatureIn
from basketball_reference_webscrapper.utils.logs import get_logger

__all__ = ['WebScrapBasketballReference']

logger = get_logger("SCRAPER_BASKETBALL_REFERENCE", log_level="INFO")

# Constants
VALID_DATA_TYPES = ["gamelog", "schedule", "player_attributes"]
MIN_SUPPORTED_SEASON = 1947
DEFAULT_REQUEST_DELAY = 20  # seconds between requests
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds between retries

# Request headers to mimic browser behavior
REQUEST_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Referer': 'https://www.basketball-reference.com/'
}


class WebScrapBasketballReference:
    """
    Class that fetches NBA data from Basketball Reference website via web scraping.

    This class provides web scraping functionality to extract NBA game data,
    schedules, and player attributes from Basketball Reference. It handles
    rate limiting, retries, and data transformation to produce clean DataFrames.

    Attributes:
        feature_object (FeatureIn): Input feature object containing data_type,
            season, and team parameters.

    Example:
        >>> from basketball_reference_webscrapper.data_models.feature_model import FeatureIn
        >>> feature = FeatureIn(data_type='gamelog', season=2023, team='BOS')
        >>> scraper = WebScrapBasketballReference(feature_object=feature)
        >>> data = scraper.fetch_basketball_reference_data()
        >>> print(data.head())
    """

    def __init__(
        self,
        feature_object: FeatureIn,
        request_delay: float = DEFAULT_REQUEST_DELAY,
        max_retries: int = MAX_RETRIES
    ) -> None:
        """
        Initialize the Basketball Reference web scraper.

        Args:
            feature_object (FeatureIn): Feature object containing data_type,
                season, and team parameters.
            request_delay (float): Delay in seconds between requests to respect
                rate limiting. Defaults to 20 seconds.
            max_retries (int): Maximum number of retry attempts for failed requests.
                Defaults to 3.
        """
        self.feature_object = feature_object
        self.request_delay = request_delay
        self.max_retries = max_retries

    def fetch_basketball_reference_data(self) -> pd.DataFrame:
        """
        Fetch NBA data from Basketball Reference via web scraping.

        This is the main entry point for fetching data. It validates inputs,
        iterates through teams, and scrapes the appropriate data based on
        the configured data_type.

        Returns:
            pd.DataFrame: DataFrame containing the scraped data with columns
                specific to the requested data_type. Returns empty DataFrame
                with expected columns if no data is found.

        Raises:
            ValueError: If data_type, season, or team parameters are invalid.

        Example:
            >>> feature = FeatureIn(data_type='gamelog', season=2023, team='BOS')
            >>> scraper = WebScrapBasketballReference(feature_object=feature)
            >>> data = scraper.fetch_basketball_reference_data()
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
        basketball_reference_data = pd.DataFrame()

        # Fetch data for each team
        for _, row in teams_to_fetch.iterrows():
            team = row["team_abrev"]
            logger.info("Fetching data for team: %s", team)

            # Build URL based on data type
            url = self._build_url(team, config)

            # Fetch and parse data with retry logic
            team_data = self._fetch_team_data_with_retry(url, team, config)

            if team_data is not None and not team_data.empty:
                team_data["id_season"] = self.feature_object.season
                team_data["tm"] = team
                basketball_reference_data = pd.concat(
                    [basketball_reference_data, team_data],
                    axis=0,
                    ignore_index=True
                )
                logger.info(
                    "Successfully fetched %d records for team: %s",
                    len(team_data),
                    team
                )
            else:
                logger.warning("No data found for team: %s", team)

            # Rate limiting delay
            time.sleep(self.request_delay)

        # Return empty DataFrame with correct columns if no data
        if basketball_reference_data.empty:
            logger.warning(
                "No data was scraped for any team. "
                "Returning empty DataFrame with expected columns."
            )
            return pd.DataFrame(
                columns=config[self.feature_object.data_type]["list_columns_to_select"]
            )

        # Select final columns
        basketball_reference_data = basketball_reference_data[
            config[self.feature_object.data_type]["list_columns_to_select"]
        ]

        logger.info(
            "Successfully fetched data: %d total records",
            len(basketball_reference_data)
        )

        return basketball_reference_data

    # Backward compatibility alias
    def webscrappe_nba_games_data(self) -> pd.DataFrame:
        """
        Fetch NBA data from Basketball Reference (deprecated).

        .. deprecated::
            Use :meth:`fetch_basketball_reference_data` instead.
            This method is kept for backward compatibility.

        Returns:
            pd.DataFrame: DataFrame containing the scraped data.
        """
        logger.warning(
            "webscrappe_nba_games_data() is deprecated. "
            "Use fetch_basketball_reference_data() instead."
        )
        return self.fetch_basketball_reference_data()

    # -------------------------------------------------------------------------
    # Private Methods - Data Fetching
    # -------------------------------------------------------------------------

    def _build_url(self, team: str, config: Dict) -> str:
        """
        Build the URL for scraping based on data type and team.

        Args:
            team (str): Team abbreviation (e.g., 'BOS').
            config (Dict): Configuration dictionary containing URL templates.

        Returns:
            str: Complete URL for the data request.
        """
        data_type = self.feature_object.data_type
        season = str(self.feature_object.season)

        if data_type == "gamelog":
            url = (
                f"{config[data_type]['url']}{team}/{season}/{data_type}"
            )
        elif data_type == "schedule":
            url = (
                f"{config[data_type]['url']}{team}/{season}_games.html"
            )
        elif data_type == "player_attributes":
            url = (
                f"{config[data_type]['url']}{team}/{season}.html"
            )
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        return url

    def _fetch_team_data_with_retry(
        self,
        url: str,
        team: str,
        config: Dict
    ) -> Optional[pd.DataFrame]:
        """
        Fetch team data with retry logic for handling transient failures.

        Args:
            url (str): URL to fetch data from.
            team (str): Team abbreviation for logging.
            config (Dict): Configuration dictionary.

        Returns:
            Optional[pd.DataFrame]: DataFrame with team data, or None if
                all retries failed.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._fetch_team_data(url, team, config)
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(
                        "Attempt %d/%d failed for team %s: %s. Retrying in %ds...",
                        attempt,
                        self.max_retries,
                        team,
                        str(e),
                        RETRY_DELAY
                    )
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(
                        "All %d attempts failed for team %s: %s",
                        self.max_retries,
                        team,
                        str(e)
                    )
                    return None

        return None

    def _fetch_team_data(
        self,
        url: str,
        team: str,
        config: Dict
    ) -> Optional[pd.DataFrame]:
        """
        Fetch and parse team data from a single URL.

        Args:
            url (str): URL to fetch data from.
            team (str): Team abbreviation for logging.
            config (Dict): Configuration dictionary.

        Returns:
            Optional[pd.DataFrame]: DataFrame with parsed team data,
                or None if request failed.

        Raises:
            Exception: If request fails or parsing errors occur.
        """
        # Check URL accessibility
        response = requests.get(
            url,
            timeout=60,
            headers=REQUEST_HEADERS
        )

        logger.info("Team %s: Status Code = %d", team, response.status_code)

        if response.status_code != 200:
            logger.warning(
                "Team %s: Got status %d for URL: %s",
                team,
                response.status_code,
                url
            )
            return None

        # Fetch and parse HTML
        req = Request(url, headers=REQUEST_HEADERS)

        with urlopen(req) as html:
            soup = BeautifulSoup(html, "html.parser")
            return self._parse_html_table(soup, config)

    def _parse_html_table(
        self,
        soup: BeautifulSoup,
        config: Dict
    ) -> Optional[pd.DataFrame]:
        """
        Parse HTML table from BeautifulSoup object.

        Args:
            soup (BeautifulSoup): Parsed HTML document.
            config (Dict): Configuration dictionary with parsing parameters.

        Returns:
            Optional[pd.DataFrame]: DataFrame with parsed data,
                or None if no data found.
        """
        data_type = self.feature_object.data_type
        data_type_config = config[data_type]

        # Extract table rows based on data type
        if data_type == 'player_attributes':
            tables = soup.findAll('table')
            if len(tables) <= data_type_config["beautifulsoup_tr_index"]:
                logger.warning("Table not found for player_attributes")
                return None
            table = tables[data_type_config["beautifulsoup_tr_index"]]
            rows = table.find_all('tr')
        else:
            rows = soup.findAll("tr")[data_type_config["beautifulsoup_tr_index"]:]

        # Extract data from rows
        rows_data = [
            [td.getText() for td in row.findAll("td")]
            for row in rows
        ]

        # Filter out empty rows
        rows_data = [row for row in rows_data if row]

        if not rows_data:
            return None

        # Create DataFrame
        df = pd.DataFrame(rows_data)
        df.columns = data_type_config["list_columns"]
        df = df.dropna()

        return df

    # -------------------------------------------------------------------------
    # Private Methods - Configuration
    # -------------------------------------------------------------------------

    def _load_config(self) -> Dict:
        """
        Load configuration from params.yaml.

        Returns:
            Dict: Configuration dictionary containing URL templates,
                column mappings, and parsing parameters.
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
                minimum supported season (1947).
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