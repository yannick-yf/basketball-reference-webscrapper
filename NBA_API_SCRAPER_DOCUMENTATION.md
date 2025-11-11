# NBA API Scraper Documentation

## Overview

The `WebScrapNBAApi` class provides an alternative data source to Basketball Reference by fetching NBA data directly from the official NBA Stats API (`stats.nba.com`). This class maintains the same interface and output structure as the `WebScrapBasketballReference` class, making it easy to switch between data sources.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Features](#features)
4. [API Reference](#api-reference)
5. [Column Mapping](#column-mapping)
6. [Supported Data Types](#supported-data-types)
7. [Limitations](#limitations)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)

## Installation

The NBA API scraper is included in the `basketball-reference-webscrapper` package and uses the official `nba_api` Python package.

```bash
pip install basketball-reference-webscrapper
# or
poetry install
```

**Dependencies:**
- `nba-api>=1.5`: Official Python wrapper for NBA Stats API

**Important:** The NBA API blocks requests from many cloud hosting providers (AWS, Heroku, Google Cloud, Digital Ocean). **This implementation will work when running locally but may fail in cloud environments.** This is a limitation of the NBA's API, not this package.

## Quick Start

```python
from basketball_reference_webscrapper.data_models.feature_model import FeatureIn
from basketball_reference_webscrapper.web_scrap_nba_api import WebScrapNBAApi

# Create feature object
feature = FeatureIn(
    data_type="gamelog",
    season=2024,
    team="BOS"
)

# Initialize scraper
scraper = WebScrapNBAApi(feature_object=feature)

# Fetch data
df = scraper.fetch_nba_api_data()

# Use the data
print(df.head())
```

## Features

### Core Features

- **Same Interface**: Uses the same `FeatureIn` data model as Basketball Reference scraper
- **Identical Output**: Returns DataFrames with matching column names and structure
- **Multiple Teams**: Supports single team, multiple teams, or all teams
- **Data Validation**: Built-in validation for data_type, season, and team inputs
- **Error Handling**: Robust exception handling with detailed logging
- **Rate Limiting**: Automatic rate limiting to respect API limits (~100 requests/minute)

### Advantages Over Web Scraping

1. **Faster**: Direct API calls are faster than web scraping
2. **More Reliable**: Less prone to breaking due to website changes
3. **Official Source**: Data comes directly from NBA's official API
4. **Real-time**: Access to current season data without delays
5. **No HTML Parsing**: Cleaner data extraction without BeautifulSoup

### Limitations

1. **Historical Data**: May have limited historical data compared to Basketball Reference
2. **Rate Limits**: Subject to NBA API rate limits
3. **Blocking**: NBA API may block requests from some cloud hosting providers (AWS, Heroku, etc.)
4. **Opponent Stats**: Some opponent statistics may not be available through certain endpoints
5. **Data Types**: Currently supports `gamelog` and `schedule` (player_attributes not yet implemented)

## API Reference

### Class: `WebScrapNBAApi`

Main class for fetching NBA data from the official API.

#### Constructor

```python
WebScrapNBAApi(feature_object: FeatureIn)
```

**Parameters:**
- `feature_object` (FeatureIn): Feature object containing:
  - `data_type` (str): Type of data to fetch (`"gamelog"` or `"schedule"`)
  - `season` (int): Season year (e.g., `2024` for 2023-24 season)
  - `team` (Union[str, List[str]]): Team abbreviation(s) or `"all"`

#### Main Method

```python
fetch_nba_api_data() -> pd.DataFrame
```

Fetches NBA data from the API and returns a DataFrame with Basketball Reference column structure.

**Returns:**
- `pd.DataFrame`: DataFrame with requested data

**Raises:**
- `ValueError`: If validation fails for data_type, season, or team

### Supported Team Abbreviations

```
ATL, BOS, BRK, CHA, CHI, CHO, CLE, DAL, DEN, DET, GSW, HOU, IND,
LAC, LAL, MEM, MIA, MIL, MIN, NOP, NYK, OKC, ORL, PHI, PHO, POR,
SAC, SAS, TOR, UTA, WAS
```

Note: Some historical teams may not be available (e.g., SEA, VAN, NJN)

## Column Mapping

### Gamelog Columns

The NBA API columns are mapped to match Basketball Reference format:

| NBA API Column | Basketball Reference Column | Description |
|----------------|----------------------------|-------------|
| GAME_DATE | game_date | Game date |
| MATCHUP | (parsed) | Used to derive extdom and opp |
| WL | results | Win/Loss |
| PTS | pts_tm | Team points |
| FGM | fg_tm | Field goals made |
| FGA | fga_tm | Field goals attempted |
| FG_PCT | fg_prct_tm | Field goal percentage |
| FG3M | 3p_tm | 3-pointers made |
| FG3A | 3pa_tm | 3-pointers attempted |
| FG3_PCT | 3p_prct_tm | 3-point percentage |
| FTM | ft_tm | Free throws made |
| FTA | fta_tm | Free throws attempted |
| FT_PCT | ft_prct_tm | Free throw percentage |
| OREB | orb_tm | Offensive rebounds |
| DREB | drb_tm | Defensive rebounds |
| REB | trb_tm | Total rebounds |
| AST | ast_tm | Assists |
| STL | stl_tm | Steals |
| BLK | blk_tm | Blocks |
| TOV | tov_tm | Turnovers |
| PF | pf_tm | Personal fouls |

**Additional Columns Added:**
- `id_season`: Season year
- `tm`: Team abbreviation
- `game_nb`: Game number (sequential)
- `extdom`: Home/Away indicator (@)
- `opp`: Opponent abbreviation

**Opponent Statistics:**
Currently, opponent statistics (fg_opp, pts_opp, etc.) are set to None/null as they require separate API calls. This is a known limitation.

### Schedule Columns

| NBA API Column | Basketball Reference Column | Description |
|----------------|----------------------------|-------------|
| GAME_DATE | game_date | Game date |
| MATCHUP | (parsed) | Used to derive extdom and opponent |
| WL | w_l | Win/Loss |
| PTS | pts_tm | Team points |

**Additional Columns Added:**
- `id_season`: Season year
- `tm`: Team abbreviation
- `time_start`: Start time (empty - not available in API)
- `extdom`: Home/Away indicator
- `opponent`: Opponent abbreviation
- `overtime`: Overtime indicator (empty - requires additional logic)
- `pts_opp`: Opponent points (null - requires additional logic)
- `w_tot`: Cumulative wins
- `l_tot`: Cumulative losses
- `streak_w_l`: Win/Loss streak (e.g., "W 3", "L 2")

## Supported Data Types

### 1. Gamelog (`data_type="gamelog"`)

Fetches detailed game-by-game statistics for a team's season.

**Endpoint Used:** `teamgamelogs`

**Available Seasons:** 2000-present (1999-00 season onwards)

**Example:**
```python
feature = FeatureIn(data_type="gamelog", season=2024, team="BOS")
```

### 2. Schedule (`data_type="schedule"`)

Fetches schedule/results with basic game information.

**Endpoint Used:** `leaguegamefinder`

**Available Seasons:** 2000-present

**Example:**
```python
feature = FeatureIn(data_type="schedule", season=2024, team="LAL")
```

### 3. Player Attributes (Not Yet Implemented)

Player attributes data type is not currently supported by the NBA API scraper. Use the Basketball Reference scraper for this data type.

## Limitations

### 1. API Rate Limits

The NBA Stats API has rate limits (~100 requests per minute). The scraper includes automatic rate limiting with 0.6-second delays between requests.

**Best Practices:**
- Avoid fetching data for "all" teams repeatedly
- Implement caching for frequently accessed data
- Use specific team queries when possible

### 2. Cloud Provider Blocking

**⚠️ CRITICAL LIMITATION**: The NBA Stats API actively blocks requests from cloud hosting providers:
- Amazon AWS ❌
- Digital Ocean ❌
- Heroku ❌
- Google Cloud Platform ❌
- Most other cloud/VPS providers ❌

**This means:**
- ✅ Works when running **locally** (home/office computer)
- ❌ **Fails in production** cloud deployments
- ❌ Fails in CI/CD pipelines (GitHub Actions, etc.)
- ❌ Fails in cloud notebooks (Google Colab, etc.)

**Error you'll see:** `JSONDecodeError: Expecting value: line 1 column 1 (char 0)` - This indicates the API returned an empty response (blocked request)

**Workarounds:**
- **Best option**: Use locally for development, use Basketball Reference scraper for production
- Run from residential IP addresses
- Use residential proxy services (not datacenter proxies)
- Note: VPNs typically don't work as most VPN IPs are also blocked

### 3. Historical Data Availability

Some historical data may not be available or may have different formats:
- Pre-2000 seasons are not supported
- Relocated/defunct teams may not have complete data
- Some statistical categories were not tracked in earlier seasons

### 4. Opponent Statistics

**Design Decision:** The NBA API scraper does **not include** opponent statistics columns (pts_opp, fg_opp, etc.) in the output. This is intentional because:
- NBA API doesn't provide opponent stats in the team gamelog/schedule endpoints
- Would require separate API calls for each game (82+ additional API calls per team)
- Significantly increased execution time and rate limit risk
- Better handled in downstream data engineering pipelines

**For Data Engineering Pipelines:**
- Fetch opponent data separately if needed
- Join/merge opponent stats in your ETL process
- This approach maintains data integrity and API efficiency

### 5. Missing Fields

Some fields available in Basketball Reference are not available in NBA API:
- Overtime indicator (requires additional logic)
- Game start time (not in API response)
- Detailed venue information

## Examples

### Example 1: Single Team Gamelog

```python
from basketball_reference_webscrapper.data_models.feature_model import FeatureIn
from basketball_reference_webscrapper.web_scrap_nba_api import WebScrapNBAApi

feature = FeatureIn(
    data_type="gamelog",
    season=2024,
    team="BOS"
)

scraper = WebScrapNBAApi(feature_object=feature)
df = scraper.fetch_nba_api_data()

print(f"Fetched {len(df)} games for Boston Celtics")
print(df[['game_date', 'opp', 'results', 'pts_tm', 'pts_opp']].head())
```

### Example 2: Multiple Teams

```python
feature = FeatureIn(
    data_type="gamelog",
    season=2023,
    team=["LAL", "BOS", "GSW"]
)

scraper = WebScrapNBAApi(feature_object=feature)
df = scraper.fetch_nba_api_data()

# Analyze by team
for team in df['tm'].unique():
    team_df = df[df['tm'] == team]
    wins = len(team_df[team_df['results'] == 'W'])
    print(f"{team}: {wins} wins")
```

### Example 3: Schedule Data

```python
feature = FeatureIn(
    data_type="schedule",
    season=2024,
    team="MIA"
)

scraper = WebScrapNBAApi(feature_object=feature)
df = scraper.fetch_nba_api_data()

# Show win/loss record over time
print(df[['game_date', 'opponent', 'w_l', 'w_tot', 'l_tot', 'streak_w_l']].head(10))
```

### Example 4: Error Handling

```python
try:
    feature = FeatureIn(
        data_type="gamelog",
        season=2024,
        team="INVALID"
    )
    scraper = WebScrapNBAApi(feature_object=feature)
    df = scraper.fetch_nba_api_data()
except ValueError as e:
    print(f"Validation error: {e}")
except Exception as e:
    print(f"API error: {e}")
```

### Example 5: All Teams (Use with Caution)

```python
# WARNING: This will make 30+ API calls and take several minutes!
feature = FeatureIn(
    data_type="schedule",
    season=2024,
    team="all"
)

scraper = WebScrapNBAApi(feature_object=feature)
df = scraper.fetch_nba_api_data()

# Analyze league-wide stats
print(f"Total games fetched: {len(df)}")
print(f"Teams: {df['tm'].nunique()}")
```

## Troubleshooting

### Problem: Import Error

```
ModuleNotFoundError: No module named 'basketball_reference_webscrapper'
```

**Solution:** Install the package:
```bash
pip install basketball-reference-webscrapper
```

### Problem: API Request Fails

```
API request failed for team BOS: 403 Client Error: Forbidden
```

**Solution:**
- Check if you're running from a blocked cloud provider
- Try running locally
- Verify your internet connection
- Wait and retry (may be temporary rate limiting)

### Problem: Empty DataFrame Returned

```
No data was fetched for any team. Returning empty DataFrame with expected columns.
```

**Solution:**
- Check if the season is valid (2000-present)
- Verify team abbreviation is correct
- Check API logs for specific errors
- Try a different team/season combination

### Problem: Slow Execution

**Solution:**
- Reduce number of teams (avoid "all" if possible)
- Implement caching mechanism
- Run during off-peak hours
- Use schedule data instead of gamelog (less data)

### Problem: Missing Opponent Stats

```
opponent statistics columns are None/null
```

**Explanation:** This is a known limitation. Opponent stats require separate API calls and are not currently implemented.

**Workaround:** Use Basketball Reference scraper for opponent statistics.

## Performance Comparison

| Feature | NBA API | Basketball Reference |
|---------|---------|---------------------|
| Speed (single team) | ~1-2 seconds | ~20-30 seconds |
| Speed (all teams) | ~30-60 seconds | ~15-20 minutes |
| Reliability | High (direct API) | Medium (web scraping) |
| Historical Data | 2000-present | 1947-present |
| Opponent Stats | Limited | Complete |
| Real-time Data | Yes | Yes (with delay) |
| Cloud Friendly | No (may be blocked) | Yes |

## Future Enhancements

Potential improvements for future versions:

1. **Opponent Statistics**: Implement optional fetching of opponent stats
2. **Caching**: Add built-in caching mechanism
3. **Retry Logic**: Implement exponential backoff for failed requests
4. **Proxy Support**: Add proxy configuration options
5. **Player Stats**: Add player-level statistics endpoints
6. **Advanced Stats**: Include advanced metrics (OffRtg, DefRtg, etc.)
7. **Playoff Data**: Support playoff games separately
8. **Team Info**: Add team roster and player information

## Support

For issues, questions, or contributions:
- GitHub Issues: [basketball-reference-webscrapper](https://github.com/yannick-yf/basketball-reference-webscrapper)
- Documentation: See main README.md

## License

Same license as the main `basketball-reference-webscrapper` package.
