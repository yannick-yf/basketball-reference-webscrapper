# basketball-reference-webscrapper

basketball-reference-webscrapper is a Python package for fetching NBA games data from **two sources**:
1. **Basketball Reference** website (web scraping)
2. **NBA Stats API** (official API via `nba_api` package)

## Features

- ✅ Web scrapes NBA gamelogs, schedules, and player attributes from Basketball Reference
- ✅ Fetches data directly from official NBA Stats API (faster, but local-only)
- ✅ Validates user inputs to ensure data accuracy
- ✅ Handles team-specific data filtering (single team, multiple teams, or all teams)
- ✅ Returns data as pandas DataFrames
- ✅ Consistent interface across both data sources

## Installation

```bash
pip install basketball-reference-webscrapper
```

**Dependencies:**
- `pandas`, `beautifulsoup4`, `requests` - for web scraping
- `nba-api` - for NBA API access (included automatically)

## Usage

### Option 1: Basketball Reference (Web Scraping)

**Best for:** Production environments, cloud deployments, historical data (1947-present)

```python
from basketball_reference_webscrapper.data_models.feature_model import FeatureIn
from basketball_reference_webscrapper.webscrapping_basketball_reference import WebScrapBasketballReference

# Create feature object
feature = FeatureIn(
    data_type='gamelog',  # 'gamelog', 'schedule', or 'player_attributes'
    season=2023,
    team='BOS'  # 'all', 'BOS', or ['BOS', 'LAL']
)

# Fetch data
scraper = WebScrapBasketballReference(feature_object=feature)
data = scraper.webscrappe_nba_games_data()
print(data.head())
```

### Option 2: NBA Stats API

**Best for:** Local development, faster data retrieval, recent seasons (2000-present)

```python
from basketball_reference_webscrapper.data_models.feature_model import FeatureIn
from basketball_reference_webscrapper.web_scrap_nba_api import WebScrapNBAApi

# Create feature object
feature = FeatureIn(
    data_type='gamelog',  # 'gamelog' or 'schedule'
    season=2023,
    team='BOS'  # 'all', 'BOS', or ['BOS', 'LAL']
)

# Fetch data
scraper = WebScrapNBAApi(feature_object=feature)
data = scraper.fetch_nba_api_data()
print(data.head())
```

**⚠️ Important:** NBA API blocks cloud providers (AWS, Heroku, GCP, etc.). Use locally only.

## Comparison: Which Data Source to Use?

| Feature | NBA API | Basketball Reference |
|---------|---------|---------------------|
| Speed | ⚡ Fast (~1-2s/team) | 🐌 Slow (~20-30s/team) |
| Cloud-friendly | ❌ No (blocks cloud IPs) | ✅ Yes |
| Historical data | 2000-present | 1947-present |
| Opponent stats | ❌ Not included | ✅ Complete |
| Player attributes | ❌ Not supported | ✅ Supported |
| Reliability | High (official API) | Medium (web scraping) |

**Recommendation:**
- **Development/Analysis (local):** Use NBA API for speed
- **Production/Cloud:** Use Basketball Reference for reliability
- **Historical research:** Use Basketball Reference
- **Need opponent stats:** Use Basketball Reference

## Supported Data Types

### Basketball Reference
- `gamelog` - Game-by-game team statistics
- `schedule` - Team schedule and results
- `player_attributes` - Player roster information

### NBA API
- `gamelog` - Game-by-game team statistics (no opponent stats)
- `schedule` - Team schedule and results (no pts_opp)

## Input Validation

Both scrapers validate inputs:

- **Data Type:** Must be valid for the chosen scraper
- **Season:** Must be integer ≥ 2000 for NBA API, ≥ 1947 for Basketball Reference
- **Team:** `'all'`, valid team abbreviation (e.g., `'BOS'`), or list of abbreviations (e.g., `['BOS', 'LAL']`)

## Valid Team Abbreviations

```
ATL, BOS, BRK, CHA, CHI, CLE, DAL, DEN, DET, GSW, HOU, IND,
LAC, LAL, MEM, MIA, MIL, MIN, NOP, NYK, OKC, ORL, PHI, PHO,
POR, SAC, SAS, TOR, UTA, WAS
```

## Examples

### Example 1: Fetch Single Team Gamelog (Basketball Reference)

```python
from basketball_reference_webscrapper.data_models.feature_model import FeatureIn
from basketball_reference_webscrapper.webscrapping_basketball_reference import WebScrapBasketballReference

feature = FeatureIn(data_type='gamelog', season=2023, team='BOS')
scraper = WebScrapBasketballReference(feature_object=feature)
data = scraper.webscrappe_nba_games_data()

print(f"Fetched {len(data)} games for Boston Celtics")
print(data[['game_date', 'opp', 'results', 'pts_tm', 'pts_opp']].head())
```

### Example 2: Fetch Multiple Teams Schedule (NBA API)

```python
from basketball_reference_webscrapper.data_models.feature_model import FeatureIn
from basketball_reference_webscrapper.web_scrap_nba_api import WebScrapNBAApi

feature = FeatureIn(data_type='schedule', season=2023, team=['LAL', 'GSW'])
scraper = WebScrapNBAApi(feature_object=feature)
data = scraper.fetch_nba_api_data()

print(f"Teams: {data['tm'].unique()}")
print(data[['game_date', 'opponent', 'w_l', 'pts_tm']].head())
```

### Example 3: Fetch All Teams (use with caution)

```python
# This will take several minutes and make 30+ requests
feature = FeatureIn(data_type='gamelog', season=2023, team='all')

# Choose your scraper based on environment
# scraper = WebScrapNBAApi(feature_object=feature)  # Local only
scraper = WebScrapBasketballReference(feature_object=feature)  # Works anywhere

data = scraper.webscrappe_nba_games_data()
print(f"Fetched data for {data['tm'].nunique()} teams")
```

## Data Engineering Use

This package is designed for data engineering pipelines:

- **Clean data structure**: Only returns actual data from source (no empty placeholders)
- **Consistent schema**: Same column names across data sources where applicable
- **Flexible filtering**: Easy to fetch specific teams or all teams
- **Error handling**: Comprehensive logging and error messages

**Note:** NBA API scraper excludes opponent statistics (would require 82+ additional API calls per team). Handle opponent data joins in your ETL pipeline if needed.

## Configuration

The package uses `params.yaml` for configuration. Both scrapers share the same team reference data in `constants/team_city_refdata.csv`.

## Troubleshooting

### NBA API: JSONDecodeError

**Error:** `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`

**Cause:** You're running in a cloud environment (AWS, Heroku, GCP, etc.). NBA API blocks datacenter IPs.

**Solution:**
- Run locally for development
- Use Basketball Reference scraper for production/cloud deployments

### Rate Limiting

- **NBA API**: ~100 requests/minute (built-in 0.4s delay between requests)
- **Basketball Reference**: Respectful delays built-in (~20s per team)

## Testing

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_web_scrap_nba_api.py -v
poetry run pytest tests/test_webscrapping_basketball_reference.py -v
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

See LICENSE file for details.

## Contact

For questions or feedback: yannick.flores1992@gmail.com

## Changelog

### v0.5.4 (Latest)
- ✨ Added NBA Stats API support via `WebScrapNBAApi` class
- ✨ Added `nba-api` package integration
- 📝 Comprehensive test coverage for both scrapers
- 🔧 Removed opponent statistics from NBA API output (data integrity)
- ⚡ Optimized rate limiting (0.4s between NBA API requests)
- 📚 Updated documentation with comparison guide

### v0.5.3
- 🐛 Fixed Basketball Reference scraper headers for better reliability
- 🔧 Improved error handling and logging

## Acknowledgments

- Basketball Reference for providing comprehensive NBA statistics
- NBA.com for the official stats API
- `nba_api` package maintainers for the excellent Python wrapper
