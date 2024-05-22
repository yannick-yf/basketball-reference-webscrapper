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

    def test_webscrappe_nba_games_data():
        webscrapping_class = WebScrapBasketballReference(
            data_type = "gamelog",
            season = 2022
        )

        nba_games = webscrapping_class.webscrappe_nba_games_data()

        assert len(nba_games) > 0

    def test_webscrappe_nba_games_data_w_wrong_data_type(self):
        """
        GIVEN a WebScrapBasketballReference object using a wrong data_type
        WHEN the webscrappe_nba_games_data method is called
        THEN check that the returned list of column names is of length 10
        """

        with self.assertRaises(ValueError):
            WebScrapBasketballReference(
                FeatureIn(
                    data_type = "data_science",
                    season = 2022)
            ).webscrappe_nba_games_data()

