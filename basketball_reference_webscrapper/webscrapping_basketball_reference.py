"""
Class that take as inputs url and season and return all games teams stats
"""

from dataclasses import dataclass
from urllib.request import urlopen
import time
from bs4 import BeautifulSoup
import pandas as pd
import requests

import importlib_resources
import yaml

from basketball_reference_webscrapper.data_models.feature_model import (
    FeatureIn
)
from basketball_reference_webscrapper.utils.logs import get_logger

logger = get_logger("WEB_SCRAPPING_EXECUTION", log_level="INFO")


@dataclass
class WebScrapBasketballReference:
    """
    Class that take as inputs url and season and return all games teams stats
    """

    def __init__(self, feature_object: FeatureIn) -> None:
        """
        Init function
        """
        self.feature_object = feature_object

    def webscrappe_nba_games_data(self):
        """
        Webscrappe NBA games data
        """

        # ------------------------------------------------------
        # User Input check and validation
        self._get_data_type_validation()
        self._get_season_validation()

        # ------------------------------------------------------
        # Import params file
        ref = (
            importlib_resources.files("basketball_reference_webscrapper")
            / "params.yaml"
        )
        with importlib_resources.as_file(ref) as path:
            with open(path, encoding="utf-8") as conf_file:
                config = yaml.safe_load(conf_file)

        # ------------------------------------------------------
        # Get team reference data
        ref = (
            importlib_resources.files("basketball_reference_webscrapper")
            / "constants/team_city_refdata.csv"
        )
        with importlib_resources.as_file(ref) as path:
            team_city_refdata = pd.read_csv(path, sep=";")

        # ------------------------------------------------------
        # Part2 - User Input check and validation
        self._get_team_list_values_validation(list(team_city_refdata["team_abrev"]))

        # ------------------------------------------------------
        # Team filtering
        if isinstance(self.feature_object.team, list):
            team_city_refdata = team_city_refdata[
                team_city_refdata["team_abrev"].isin(self.feature_object.team)
            ]
        elif isinstance(self.feature_object.team, str):
            if self.feature_object.team != "all":
                team_city_refdata = team_city_refdata[
                    team_city_refdata["team_abrev"] == self.feature_object.team
                ]

        # ------------------------------------------------------
        # Initialization of the dataframe to fill-in
        basketbal_reference_data = pd.DataFrame()

        # ------------------------------------------------------
        # For Loop Throught all the team abrev for the given season
        for index, row in team_city_refdata.iterrows():

            # URL to scrape
            team = row["team_abrev"]

            logger.info("Execution for: %s", team)

            if self.feature_object.data_type == "gamelog":
                url = (
                    config[self.feature_object.data_type]["url"]
                    + team
                    + "/"
                    + str(self.feature_object.season)
                    + "/"
                    + self.feature_object.data_type
                )
            elif self.feature_object.data_type == "schedule":
                url = (
                    config[self.feature_object.data_type]["url"]
                    + team
                    + "/"
                    + str(self.feature_object.season)
                    + "_games.html"
                )
            elif self.feature_object.data_type == "player_attributes":
                url = (
                    config[self.feature_object.data_type]["url"]
                    + team
                    + "/"
                    + str(self.feature_object.season)
                    + ".html"
                )

            try:
                response = requests.get(url, timeout=60)
                logger.info(f"Team {team}: Status Code = {response.status_code}")
                
                if response.status_code == 200:
                    # ... scraping code ...
                    # collect HTML data and create beautiful soup object:
                    with urlopen(url) as html:

                        # create beautiful soup object from HTML
                        soup = BeautifulSoup(html, "html.parser")

                        if self.feature_object.data_type == 'player_attributes':
                            rows = soup.findAll('table')[config[self.feature_object.data_type]["beautifulsoup_tr_index"]]
                            rows = rows.find_all('tr')
                        else:
                            rows = soup.findAll("tr")[
                                config[self.feature_object.data_type]["beautifulsoup_tr_index"] :
                            ]

                        rows_data = [
                            [td.getText() for td in rows[i].findAll("td")]
                            for i in range(len(rows))
                        ]

                        if len(rows_data) != 0:

                            basketbal_reference_data_tmp = pd.DataFrame(rows_data)
                            basketbal_reference_data_tmp.columns = config[self.feature_object.data_type]["list_columns"]
                            basketbal_reference_data_tmp = basketbal_reference_data_tmp.dropna()
                            basketbal_reference_data_tmp.loc[:, "id_season"] = self.feature_object.season
                            basketbal_reference_data_tmp.loc[:, "tm"] = team
                            basketbal_reference_data = pd.concat([basketbal_reference_data, basketbal_reference_data_tmp], axis=0)

                    ################################################
                else:
                    logger.warning(f"Team {team}: Got status {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Team {team}: Exception - {str(e)}")

            # if "200" in str(requests.get(url, timeout=60)):
                
            #     # collect HTML data and create beautiful soup object:
            #     with urlopen(url) as html:

            #         # create beautiful soup object from HTML
            #         soup = BeautifulSoup(html, "html.parser")

            #         if self.feature_object.data_type == 'player_attributes':
            #             rows = soup.findAll('table')[config[self.feature_object.data_type]["beautifulsoup_tr_index"]]
            #             rows = rows.find_all('tr')
            #         else:
            #             rows = soup.findAll("tr")[
            #                 config[self.feature_object.data_type]["beautifulsoup_tr_index"] :
            #             ]

            #         rows_data = [
            #             [td.getText() for td in rows[i].findAll("td")]
            #             for i in range(len(rows))
            #         ]

            #         if len(rows_data) != 0:

            #             basketbal_reference_data_tmp = pd.DataFrame(rows_data)
            #             basketbal_reference_data_tmp.columns = config[self.feature_object.data_type]["list_columns"]
            #             basketbal_reference_data_tmp = basketbal_reference_data_tmp.dropna()
            #             basketbal_reference_data_tmp.loc[:, "id_season"] = self.feature_object.season
            #             basketbal_reference_data_tmp.loc[:, "tm"] = team
            #             basketbal_reference_data = pd.concat([basketbal_reference_data, basketbal_reference_data_tmp], axis=0)

            time.sleep(20)

        basketbal_reference_data = basketbal_reference_data[config[self.feature_object.data_type]["list_columns_to_select"]]

        return basketbal_reference_data

    def _get_data_type_validation(self):
        if self.feature_object.data_type is not None:
            if str(self.feature_object.data_type) not in ["gamelog", "schedule", "player_attributes"]:
                raise ValueError(
                    "data_type value provided is not supported by the package.\
                    Accepted values are: 'gamelog', 'schedule'.\
                    Read documentation for more details"
                )
        else:
            raise ValueError(
                "data_type is a required argument.\
                    Accepted values are: 'gamelog', 'schedule', 'player_attributes'.\
                    Read documentation for more details"
            )

    def _get_season_validation(self):
        if isinstance(self.feature_object.season, int):
            if self.feature_object.season < 1999:
                raise ValueError(
                    "season value provided is a year not supported by the package. \
                    it should be between 2000 and current NBA season."
                )
        else:
            raise ValueError(
                "season is a required argument and should be an int value between 2000\
                    and current NBA season."
            )

    def _get_team_list_values_validation(self, team_abbrev_list: list):
        if (isinstance(self.feature_object.team, list)) and (
            all(isinstance(s, str) for s in self.feature_object.team)
        ):
            if set(self.feature_object.team).issubset(team_abbrev_list) is False:
                raise ValueError(
                    "team list arg provided is not accepted.\
                    Value needs to be 'all' or a NBA team abbreviation\
                    such as BOS for Boston Celtics.\
                    More details on all values possible in the GitHub Repo docs"
                )
        elif isinstance(self.feature_object.team, str):
            if (self.feature_object.team != "all") and (
                self.feature_object.team not in team_abbrev_list
            ):
                raise ValueError(
                    "team arg provided is not accepted.\
                    Value needs to be 'all' or a NBA team abbreviation\
                    such as BOS for Boston Celtics.\
                    More details on all values possible in the GitHub Repo docs"
                )
        else:
            raise ValueError(
                "team args should be a string or a list of string.\
                    Value needs to be NBA team abbreviation such as BOS for Boston Celtics.\
                    More details on all values possible in the GitHub Repo docs"
            )
