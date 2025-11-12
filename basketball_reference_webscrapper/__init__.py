"""
Basketball Reference Web Scrapper Package

This package provides tools to scrape NBA basketball data from multiple sources:
- Basketball Reference website (web scraping)
- NBA Stats API (official API)
"""

__version__ = "0.5.3"

from basketball_reference_webscrapper.webscrapping_basketball_reference import (
    WebScrapBasketballReference
)
from basketball_reference_webscrapper.web_scrap_nba_api import WebScrapNBAApi
from basketball_reference_webscrapper.data_models.feature_model import FeatureIn

__all__ = [
    "WebScrapBasketballReference",
    "WebScrapNBAApi",
    "FeatureIn",
]
