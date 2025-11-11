# NBA API Scraper - Quick Start

## Overview

The `WebScrapNBAApi` class provides an alternative to Basketball Reference web scraping by using the official NBA Stats API through the `nba_api` Python package.

## ⚠️ Important Limitations

**Cloud Provider Blocking:** The NBA Stats API blocks requests from cloud hosting providers (AWS, Heroku, GCP, etc.). This implementation:
- ✅ **Works locally** (your computer)
- ❌ **Fails in cloud** deployments

For production use in cloud environments, use the Basketball Reference scraper instead.

## Installation

The `nba-api` package is included as a dependency:

```bash
pip install basketball-reference-webscrapper
```

## Basic Usage

```python
from basketball_reference_webscrapper.data_models.feature_model import FeatureIn
from basketball_reference_webscrapper.web_scrap_nba_api import WebScrapNBAApi

# Create feature object (same as Basketball Reference scraper)
feature = FeatureIn(
    data_type="gamelog",  # or "schedule"
    season=2023,
    team="BOS"  # or ["BOS", "LAL"] or "all"
)

# Initialize scraper
scraper = WebScrapNBAApi(feature_object=feature)

# Fetch data
df = scraper.fetch_nba_api_data()

# Use the data (same columns as Basketball Reference)
print(df.head())
```

## Supported Data Types

- `gamelog`: Game-by-game team statistics
- `schedule`: Schedule and results with win/loss records

## Comparison with Basketball Reference

| Feature | NBA API | Basketball Reference |
|---------|---------|---------------------|
| Speed | Fast (~1-2s per team) | Slow (~20-30s per team) |
| Cloud-friendly | ❌ No | ✅ Yes |
| Historical data | 2000-present | 1947-present |
| Opponent stats | Limited | Complete |
| Reliability | High (official API) | Medium (web scraping) |

## Key Differences from Raw API

This implementation uses the `nba_api` Python package which:
- Handles API authentication and headers automatically
- Provides proper endpoint wrappers
- Manages request formatting
- Converts responses to pandas DataFrames

## When to Use Each Scraper

### Use NBA API Scraper when:
- Running locally for development/analysis
- You need fast data retrieval
- Working with recent seasons (2000+)
- You don't need opponent statistics

### Use Basketball Reference Scraper when:
- Deploying to cloud environments
- You need historical data (pre-2000)
- You need complete opponent statistics
- Cloud reliability is important

## Complete Documentation

See [NBA_API_SCRAPER_DOCUMENTATION.md](NBA_API_SCRAPER_DOCUMENTATION.md) for:
- Detailed API reference
- Column mapping
- Advanced examples
- Troubleshooting guide
- Performance comparison

## Examples

See [example_nba_api_usage.py](example_nba_api_usage.py) for working examples.

## Support

- GitHub: [basketball-reference-webscrapper](https://github.com/yannick-yf/basketball-reference-webscrapper)
- Issues: Report NBA API specific issues with `[NBA API]` tag
