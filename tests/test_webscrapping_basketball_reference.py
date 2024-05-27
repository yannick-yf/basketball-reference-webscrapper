"""Test WebScrapBasketballReference"""

from unittest import TestCase
from basketball_reference_webscrapper.webscrapping_basketball_reference import WebScrapBasketballReference
from basketball_reference_webscrapper.data_models.feature_model import FeatureIn

class TestWebScrapBasketballReference(TestCase):
    def setUp(self) -> None:
        self.n_samples = 500
        self.n_features=20
        self.noise=0.1
        self.n_informative=5
        self.random_state=42

    def test_webscrappe_nba_games_data(self):
        """
        GIVEN a WebScrapBasketballReference object with right arguments
            and the default value for team
        WHEN the webscrappe_nba_games_data method is called
        THEN it returns a valid results
        """

        nba_games = WebScrapBasketballReference(
                FeatureIn(
                    data_type = "gamelog",
                    season = 2022)
            ).webscrappe_nba_games_data()

        assert len(nba_games) > 0

    def test_webscrappe_nba_games_data_with_two_teams_list_arg(self):
        """
        GIVEN a WebScrapBasketballReference object with right arguments 
            and a list of two teams for the argument team
        WHEN the webscrappe_nba_games_data method is called
        THEN it returns a valid results
        """

        nba_games = WebScrapBasketballReference(
                FeatureIn(
                    data_type = "gamelog",
                    season = 2022,
                    team = ['BOS', 'LAL'])
            ).webscrappe_nba_games_data()

        assert len(nba_games) > 0

    def test_webscrappe_nba_games_data_with_one_team_arg(self):
        """
        GIVEN a WebScrapBasketballReference object with right arguments
            and a string value of one team for the argument team
        WHEN the webscrappe_nba_games_data method is called
        THEN it returns a valid results
        """

        nba_games = WebScrapBasketballReference(
                FeatureIn(
                    data_type = "gamelog",
                    season = 2022,
                    team = 'BOS')
            ).webscrappe_nba_games_data()

        assert len(nba_games) > 0

    def test_webscrappe_nba_games_data_w_wrong_team_value(self):
        """
        GIVEN a WebScrapBasketballReference object using a wrong team value
        WHEN the webscrappe_nba_games_data method is called
        THEN check that the returned an error
        """

        with self.assertRaises(ValueError):
            WebScrapBasketballReference(
                FeatureIn(
                    data_type = "gamelog",
                    season = 2022,
                    team='TOTO')
            ).webscrappe_nba_games_data()

    def test_webscrappe_nba_games_data_w_wrong_team_list_value(self):
        """
        GIVEN a WebScrapBasketballReference object using a wrong team value
        WHEN the webscrappe_nba_games_data method is called
        THEN check that the returned an error
        """

        with self.assertRaises(ValueError):
            WebScrapBasketballReference(
                FeatureIn(
                    data_type = "gamelog",
                    season = 2022,
                    team=['TOTO', 'ATL'])
            ).webscrappe_nba_games_data()

    def test_webscrappe_nba_games_data_w_wrong_data_type(self):
        """
        GIVEN a WebScrapBasketballReference object using a wrong data_type
        WHEN the webscrappe_nba_games_data method is called
        THEN check that the returned lan error
        """

        with self.assertRaises(ValueError):
            WebScrapBasketballReference(
                FeatureIn(
                    data_type = "data_science",
                    season = 2022)
            ).webscrappe_nba_games_data()

    def test_webscrappe_nba_games_data_w_wrong_season(self):
        """
        GIVEN a WebScrapBasketballReference object using a string as season variable
        WHEN the webscrappe_nba_games_data method is called
        THEN return an error that the season used is not at the accepted format
        """

        with self.assertRaises(ValueError):
            WebScrapBasketballReference(
                FeatureIn(
                    data_type = "gamelog",
                    season = "1234")
            ).webscrappe_nba_games_data()

    def test_webscrappe_nba_games_data_w_non_supported_season(self):
        """
        GIVEN a WebScrapBasketballReference object using a NBA season before 2000
        WHEN the webscrappe_nba_games_data method is called
        THEN return an error that the season used is before 2000
        """

        with self.assertRaises(ValueError):
            WebScrapBasketballReference(
                FeatureIn(
                    data_type = "gamelog",
                    season = 1998)
            ).webscrappe_nba_games_data()

