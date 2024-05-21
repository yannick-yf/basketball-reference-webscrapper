"""Test WebScrapBasketballReference"""

import pytest
from basketball_reference_webscrapper.webscrapping_basketball_reference import WebScrapBasketballReference

def test_webscrappe_nba_games_data():

    webscrapping_class = WebScrapBasketballReference(
        url = "test",
        season = 2022
    )

    nba_games = webscrapping_class.webscrappe_nba_games_data()

    assert len(nba_games) > 0

