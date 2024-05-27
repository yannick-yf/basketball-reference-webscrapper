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
    
    def __init__(self, feature_object: FeatureIn) -> None:
        self.feature_object = feature_object
    
    def _get_team_list_values_validation(self, team_abbrev_list: list):
        if  (isinstance(self.feature_object.team, list)) and (all(isinstance(s, str) for s in self.feature_object.team)):
            if set(self.feature_object.team).issubset(team_abbrev_list)==False:
                raise ValueError(
                    "team list arg provided is not accepted.\
                    Value needs to be 'all' or a NBA team abbreviation such as BOS for Boston Celtics.\
                    More details on all values possible in the GitHub Repo docs"
                )
        elif isinstance(self.feature_object.team, str):
            if (self.feature_object.team != 'all') and (self.feature_object.team not in team_abbrev_list):
                raise ValueError(
                    "team arg provided is not accepted.\
                    Value needs to be 'all' or a NBA team abbreviation such as BOS for Boston Celtics.\
                    More details on all values possible in the GitHub Repo docs"
                )
        else:
            raise ValueError(
                    "team args should be a string or a list of string.\
                    Value needs to be NBA team abbreviation such as BOS for Boston Celtics.\
                    More details on all values possible in the GitHub Repo docs"
                )

    def webscrappe_nba_games_data(self):
        """
        Webscrappe NBA games data
        """

        #------------------------------------------------------
        # Get team reference data 
        ref = (
            importlib_resources.files("basketball_reference_webscrapper")
            / "constants/team_city_refdata.csv"
        )
        with importlib_resources.as_file(ref) as path:
            team_city_refdata = pd.read_csv(path, sep = ';')

        #------------------------------------------------------
        # User Input check and validation
        self._get_data_type_validation()
        self._get_season_validation()
        self._get_team_list_values_validation(list(team_city_refdata['team_abrev']))

        #------------------------------------------------------
        # Team filtering
        if isinstance(self.feature_object.team, list):
            team_city_refdata = team_city_refdata[team_city_refdata['team_abrev'].isin(self.feature_object.team)]
        elif isinstance(self.feature_object.team, str):
            if self.feature_object.team != 'all':
                team_city_refdata = team_city_refdata[team_city_refdata['team_abrev'] == self.feature_object.team]
        
        #------------------------------------------------------
        # Initialization of the dataframe to fill-in
        games = pd.DataFrame()

        #------------------------------------------------------
        # For Loop Throught all the team abrev for the given season
        for index, row in team_city_refdata.iterrows():

            # URL to scrape
            team = row['team_abrev']

            logger.info(
                f"Execution for {team}"
            )
            
            if self.feature_object.data_type == 'gamelog':
                url = "https://www.basketball-reference.com/teams/" +\
                    team + "/" +\
                    str(self.feature_object.season) +\
                    "/"+\
                    self.feature_object.data_type
            else:
                url = "https://www.basketball-reference.com/teams/ATL/2022_games.html"

            # url = f'https://www.basketball-reference.com/players/{player_id}/gamelog/{id_season}'
            # url = f"https://www.basketball-reference.com/teams/{team}/{SEASON}.html"
            # # url = f"https://www.basketball-reference.com/teams/BRK/2001.html"

            if '200' in str(requests.get(url)):

                # collect HTML data and create beautiful soup object:
                html = urlopen(url)
                        
                # create beautiful soup object from HTML
                soup = BeautifulSoup(html, "html.parser")


                if self.feature_object.data_type == 'gamelog':
                    rows = soup.findAll('tr')[2:]
                else:
                    rows = soup.findAll('tr')[1:]

                rows_data = [[td.getText() for td in rows[i].findAll('td')]
                                    for i in range(len(rows))]

                if len(rows_data) != 0:
                    # create the dataframe
                    games_tmp = pd.DataFrame(rows_data)

                    if self.feature_object.data_type == 'gamelog':
                        cols = ["game_nb", "game_date", "extdom", "opp", "results",
                            "pts_tm","pts_opp",
                            "fg_tm", "fga_tm","fg_prct_tm",
                            "3p_tm","3pa_tm", "3p_prct_tm","ft_tm","fta_tm","ft_prct_tm",
                            "orb_tm","trb_tm", "ast_tm","stl_tm","blk_tm" ,"tov_tm","pf_tm",
                            "nc",
                            "fg_opp","fga_opp","fg_prct_opp",
                            "3p_opp", "3pa_opp", "3p_prct_opp", "ft_opp", "fta_opp","ft_prct_opp",
                            "orb_opp", "trb_opp","ast_opp", "stl_opp", "blk_opp","tov_opp", "pf_opp"]
                    else:
                        cols = ['game_date', 'time_start', 'nc1', 'nc2', 'extdom', 'opponent', 'w_l', 'overtime', 'pts_tm', 'pts_opp', 'w_tot', 'l_tot', 'streak_w_l', 'nc3']
                    
                    games_tmp.columns =  cols
                    games_tmp = games_tmp.dropna()
                    games_tmp['id_season'] = self.feature_object.season
                    games_tmp['tm'] = team
                    games = pd.concat([games, games_tmp], axis=0)
            
            time.sleep(5)

        if self.feature_object.data_type == 'gamelog':
            games = games[[
                'id_season', 'game_nb', 'game_date', 'extdom', 'tm','opp', 'results', 'pts_tm', 'pts_opp',
                'fg_tm', 'fga_tm', 'fg_prct_tm', '3p_tm', '3pa_tm', '3p_prct_tm',
                'ft_tm', 'fta_tm', 'ft_prct_tm', 'orb_tm', 'trb_tm', 'ast_tm', 'stl_tm',
                'blk_tm', 'tov_tm', 'pf_tm', 'fg_opp', 'fga_opp', 'fg_prct_opp',
                '3p_opp', '3pa_opp', '3p_prct_opp', 'ft_opp', 'fta_opp', 'ft_prct_opp',
                'orb_opp', 'trb_opp', 'ast_opp', 'stl_opp', 'blk_opp', 'tov_opp',
                'pf_opp']]
        else:
            games = games[['id_season', 'tm', 'game_date', 'time_start', 'extdom', 'opponent', 'w_l', 'overtime', 'pts_tm', 'pts_opp', 'w_tot', 'l_tot', 'streak_w_l']]
            
        return games
    

    def _get_data_type_validation(self):
        if self.feature_object.data_type is not None:
            if str(self.feature_object.data_type) not in ['gamelog', 'schedule']:
                raise ValueError(
                    "data_type value provided is not supported by the package.\
                    Accepted values are: 'gamelog', 'schedule'.\
                    Read documentation for more details"
                )
        else:
            raise ValueError(
                    "data_type is a required argument.\
                    Accepted values are: 'gamelog', 'schedule'.\
                    Read documentation for more details"
                )
    
    def _get_season_validation(self):
        if isinstance(self.feature_object.season, int):
            if (
                self.feature_object.season < 1999
            ):
                raise ValueError(
                    "season value provided is a year not supported by the package. \
                    it should be between 2000 and current NBA season."
                )
        else:
            raise ValueError(
                    "season is a required argument and should be an int value between 2000 and current NBA season."
                )
    

    
if __name__ == "__main__":
    webscrapping_class = WebScrapBasketballReference(
        url = "test",
        season = 2022
    )

    nba_games = webscrapping_class.webscrappe_nba_games_data()

