"""Test WebScrapBasketballReference"""

from unittest import TestCase
from basketball_reference_webscrapper.web_scrap_basketball_reference import (
    WebScrapBasketballReference,
)
from basketball_reference_webscrapper.data_models.feature_model import FeatureIn


class TestWebScrapBasketballReference(TestCase):
    def setUp(self) -> None:
        self.n_samples = 500
        self.n_features = 20
        self.noise = 0.1
        self.n_informative = 5
        self.random_state = 42

    # def test_fetch_basketball_reference_data(self):
    #     """
    #     GIVEN a WebScrapBasketballReference object with right arguments
    #         and the default value for team
    #     WHEN the fetch_basketball_reference_data method is called
    #     THEN it returns a valid results
    #     """

    #     nba_games = WebScrapBasketballReference(
    #         FeatureIn(data_type="gamelog", season=2018)
    #     ).fetch_basketball_reference_data()

    #     assert len(nba_games) > 0

    def test_webscrappe_nba_player_attributes(self):
        """
        GIVEN a WebScrapBasketballReference object with right arguments
            with team args with only one team
            with data_type
        WHEN the fetch_basketball_reference_data method is called
        THEN it returns a valid results
        """

        nba_games = WebScrapBasketballReference(
            FeatureIn(data_type="player_attributes", season=2010, team="BOS")
        ).fetch_basketball_reference_data()

        assert len(nba_games) > 0

    def test_fetch_basketball_reference_data_schedule(self):
        """
        GIVEN a WebScrapBasketballReference object with right arguments
            and a list of two teams for the argument team
        WHEN the fetch_basketball_reference_data method is called
        THEN it returns a valid results
        """

        nba_games = WebScrapBasketballReference(
            FeatureIn(data_type="schedule", season=2022, team="BOS")
        ).fetch_basketball_reference_data()

        assert len(nba_games) > 0

    def test_fetch_basketball_reference_data_schedule_two_teams_list_arg(self):
        """
        GIVEN a WebScrapBasketballReference object with right arguments
            and a list of two teams for the argument team
        WHEN the fetch_basketball_reference_data method is called
        THEN it returns a valid results
        """

        nba_games = WebScrapBasketballReference(
            FeatureIn(data_type="schedule", season=2010, team=["BOS", "DAL"])
        ).fetch_basketball_reference_data()

        assert len(nba_games) > 0

    def test_fetch_basketball_reference_data_with_two_teams_list_arg(self):
        """
        GIVEN a WebScrapBasketballReference object with right arguments
            and a list of two teams for the argument team
        WHEN the fetch_basketball_reference_data method is called
        THEN it returns a valid results
        """

        nba_games = WebScrapBasketballReference(
            FeatureIn(data_type="gamelog", season=2022, team=["BOS", "LAL"])
        ).fetch_basketball_reference_data()

        assert len(nba_games) > 0

    def test_fetch_basketball_reference_data_with_one_team_arg(self):
        """
        GIVEN a WebScrapBasketballReference object with right arguments
            and a string value of one team for the argument team
        WHEN the fetch_basketball_reference_data method is called
        THEN it returns a valid results
        """

        nba_games = WebScrapBasketballReference(
            FeatureIn(data_type="gamelog", season=2022, team="BOS")
        ).fetch_basketball_reference_data()

        assert len(nba_games) > 0

    def test_fetch_basketball_reference_data_w_wrong_team_value(self):
        """
        GIVEN a WebScrapBasketballReference object using a wrong team value
        WHEN the fetch_basketball_reference_data method is called
        THEN check that the returned an error
        """

        with self.assertRaises(ValueError):
            WebScrapBasketballReference(
                FeatureIn(data_type="gamelog", season=2022, team="TOTO")
            ).fetch_basketball_reference_data()

    def test_fetch_basketball_reference_data_w_wrong_team_list_value(self):
        """
        GIVEN a WebScrapBasketballReference object using a wrong team value
        WHEN the fetch_basketball_reference_data method is called
        THEN check that the returned an error
        """

        with self.assertRaises(ValueError):
            WebScrapBasketballReference(
                FeatureIn(data_type="gamelog", season=2022, team=["TOTO", "ATL"])
            ).fetch_basketball_reference_data()

    def test_fetch_basketball_reference_data_w_wrong_data_type(self):
        """
        GIVEN a WebScrapBasketballReference object using a wrong data_type
        WHEN the fetch_basketball_reference_data method is called
        THEN check that the returned lan error
        """

        with self.assertRaises(ValueError):
            WebScrapBasketballReference(
                FeatureIn(data_type="data_science", season=2022)
            ).fetch_basketball_reference_data()

    def test_fetch_basketball_reference_data_w_wrong_season(self):
        """
        GIVEN a WebScrapBasketballReference object using a string as season variable
        WHEN the fetch_basketball_reference_data method is called
        THEN return an error that the season used is not at the accepted format
        """

        with self.assertRaises(ValueError):
            WebScrapBasketballReference(
                FeatureIn(data_type="gamelog", season="1234")
            ).fetch_basketball_reference_data()

    def test_fetch_basketball_reference_data_w_non_supported_season(self):
        """
        GIVEN a WebScrapBasketballReference object using a NBA season before 2000
        WHEN the fetch_basketball_reference_data method is called
        THEN return an error that the season used is before 2000
        """

        with self.assertRaises(ValueError):
            WebScrapBasketballReference(
                FeatureIn(data_type="gamelog", season=1998)
            ).fetch_basketball_reference_data()
