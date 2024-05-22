from dataclasses import dataclass
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import requests
import time
import importlib_resources

from basketball_reference_webscrapper.data_models.feature_model import FeatureIn, FeatureOut
from basketball_reference_webscrapper.utils.logs import get_logger

logger = get_logger("WEB_SCRAPPING_EXECUTION", log_level="INFO")

@dataclass
class WebScrapBasketballReference:
    """
    Class that take as inputs url and season and return all games teams stats
    """

    # init method or constructor
    # def __init__(self, url, season):
    #     self.url = url
    #     self.season = season
    
    def __init__(self, feature_object: FeatureIn) -> None:
        self.feature_object = feature_object

    def webscrappe_nba_games_data(self):
        """
        Webscrappe NBA games data
        """

        self._get_data_type_validation()
        self._get_season_validation()
        
        #------------------------------------------------------
        # Get team reference data 
        ref = (
            importlib_resources.files("basketball_reference_webscrapper")
            / "constants/team_city_refdata.csv"
        )
        with importlib_resources.as_file(ref) as path:
            # Do something with path.  After the with-statement exits, any
            # temporary file created will be immediately cleaned up.
            team_city_refdata = pd.read_csv(path, sep = ';')

        #------------------------------------------------------
        # Initialization of the dataframe to fill-in
        games = pd.DataFrame()

        #------------------------------------------------------
        # For Loop Throught all the team abrev for the given season

        team_city_refdata = team_city_refdata.head(2)

        for index, row in team_city_refdata.iterrows():

            # URL to scrape
            team = row['team_abrev']

            logger.info(
                f"Execution for {team}"
            )
            
            url = "https://www.basketball-reference.com/teams/ATL/" +\
                self.feature_object.data_type +\
                "/" +\
                str(self.feature_object.season) +\
                "/gamelog/"
            
            # url = f"https://www.basketball-reference.com/teams/{team}/{self.season}/gamelog/"
            # url = f"https://www.basketball-reference.com/teams/ATL/2022/gamelog/"

            print(url)

            if '200' in str(requests.get(url)):

                # collect HTML data and create beautiful soup object:
                # collect HTML data
                html = urlopen(url)
                        
                # create beautiful soup object from HTML
                soup = BeautifulSoup(html, "html.parser")

                rows = soup.findAll('tr')[2:]

                rows_data = [[td.getText() for td in rows[i].findAll('td')]
                                    for i in range(len(rows))]

                if len(rows_data) != 0:
                    # create the dataframe
                    games_tmp = pd.DataFrame(rows_data)
                    cols = ["game_nb", "game_date", "extdom", "opp", "results",
                            "pts_tm","pts_opp",
                            "fg_tm", "fga_tm","fg_prct_tm",
                            "3p_tm","3pa_tm", "3p_prct_tm","ft_tm","fta_tm","ft_prct_tm",
                            "orb_tm","trb_tm", "ast_tm","stl_tm","blk_tm" ,"tov_tm","pf_tm",
                            "nc",
                            "fg_opp","fga_opp","fg_prct_opp",
                            "3p_opp", "3pa_opp", "3p_prct_opp", "ft_opp", "fta_opp","ft_prct_opp",
                            "orb_opp", "trb_opp","ast_opp", "stl_opp", "blk_opp","tov_opp", "pf_opp"]

                    games_tmp.columns =  cols
                    games_tmp = games_tmp.dropna()
                    games_tmp['id_season'] = self.season
                    games_tmp['tm'] = team
                    games = pd.concat([games, games_tmp], axis=0)
            
            time.sleep(5)

        games = games[[
            'id_season', 'game_nb', 'game_date', 'extdom', 'tm','opp', 'results', 'pts_tm', 'pts_opp',
            'fg_tm', 'fga_tm', 'fg_prct_tm', '3p_tm', '3pa_tm', '3p_prct_tm',
            'ft_tm', 'fta_tm', 'ft_prct_tm', 'orb_tm', 'trb_tm', 'ast_tm', 'stl_tm',
            'blk_tm', 'tov_tm', 'pf_tm', 'fg_opp', 'fga_opp', 'fg_prct_opp',
            '3p_opp', '3pa_opp', '3p_prct_opp', 'ft_opp', 'fta_opp', 'ft_prct_opp',
            'orb_opp', 'trb_opp', 'ast_opp', 'stl_opp', 'blk_opp', 'tov_opp',
            'pf_opp']]

        return games
    

    def _get_data_type_validation(self):
        if self.feature_object.data_type is not None:
            print(self.feature_object.data_type)
            if str(self.feature_object.data_type) != 'gamelog':
                raise ValueError(
                    "data_type value provided is not supported by the package. Accepted values are: 'gamelog'\
                    Read documentation for more details"
                )
        else:
            raise ValueError(
                    "data_type is a required argument. it accepts the following values: 'gamelog'.\
                    Read documentation for more details"
                )
    
    def _get_season_validation(self):
        if self.feature_object.season is not None:
            if (
                self.feature_object.season > 1999
            ):
                raise ValueError(
                    "season value provided is a year not supported by the package. \
                    it should be between 2000 and current NBA season."
                )
        else:
            raise ValueError(
                    "season is a required argument. it should be between 2000 and current NBA season."
                )
    

    
if __name__ == "__main__":
    webscrapping_class = WebScrapBasketballReference(
        url = "test",
        season = 2022
    )

    nba_games = webscrapping_class.webscrappe_nba_games_data()

