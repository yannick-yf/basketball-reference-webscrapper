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
from basketball_reference_webscrapper.data_models.feature_model import FeatureIn

# Optional NBA API import - won't fail if nba_api is not installed or has issues
try:
    from basketball_reference_webscrapper.web_scrap_nba_api import WebScrapNBAApi
    _NBA_API_AVAILABLE = True
except ImportError as e:
    WebScrapNBAApi = None
    _NBA_API_AVAILABLE = False
    import warnings
    warnings.warn(
        f"NBA API scraper not available: {e}. "
        "Install nba-api package to use WebScrapNBAApi. "
        "Basketball Reference scraper still works normally.",
        ImportWarning
    )

__all__ = [
    "WebScrapBasketballReference",
    "WebScrapNBAApi",
    "FeatureIn",
]
