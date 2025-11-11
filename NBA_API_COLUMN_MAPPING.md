# NBA API to Basketball Reference Column Mapping

This document provides detailed mapping information for data transformation between NBA Stats API and Basketball Reference format.

## Overview

The `WebScrapNBAApi` class transforms NBA Stats API responses to match Basketball Reference column names and formats. This ensures compatibility and allows users to switch between data sources seamlessly.

## Gamelog Column Mapping

### Complete Mapping Table

| Basketball Reference Column | NBA API Source Column | Data Type | Transformation Notes |
|----------------------------|----------------------|-----------|---------------------|
| **Metadata Columns** |
| id_season | (added) | int | Season year from input parameter |
| tm | (added) | str | Team abbreviation from input parameter |
| game_nb | (calculated) | int | Sequential game number (1, 2, 3, ...) |
| game_date | GAME_DATE | str | Format: "Mon, Oct 23, 2023" or varies by API response |
| extdom | MATCHUP (parsed) | str | "@" if away game, "" if home; parsed from "vs" or "@" in MATCHUP |
| opp | MATCHUP (parsed) | str | Opponent abbreviation; extracted from MATCHUP string |
| results | WL | str | "W" for win, "L" for loss |
| **Scoring Columns** |
| pts_tm | PTS | int | Total points scored by team |
| pts_opp | (not available) | int/null | Currently set to None; requires separate API call |
| **Field Goals** |
| fg_tm | FGM | int | Field goals made |
| fga_tm | FGA | int | Field goals attempted |
| fg_prct_tm | FG_PCT | str | Formatted as "0.XXX" (e.g., "0.485") |
| **3-Point Shooting** |
| 3p_tm | FG3M | int | 3-pointers made |
| 3pa_tm | FG3A | int | 3-pointers attempted |
| 3p_prct_tm | FG3_PCT | str | Formatted as "0.XXX" |
| **Free Throws** |
| ft_tm | FTM | int | Free throws made |
| fta_tm | FTA | int | Free throws attempted |
| ft_prct_tm | FT_PCT | str | Formatted as "0.XXX" |
| **Rebounds** |
| orb_tm | OREB | int | Offensive rebounds |
| drb_tm | DREB | int | Defensive rebounds (not in BR gamelog) |
| trb_tm | REB | int | Total rebounds |
| **Other Stats** |
| ast_tm | AST | int | Assists |
| stl_tm | STL | int | Steals |
| blk_tm | BLK | int | Blocks |
| tov_tm | TOV | int | Turnovers |
| pf_tm | PF | int | Personal fouls |
| **Opponent Stats** |
| fg_opp | (not available) | int/null | Currently None |
| fga_opp | (not available) | int/null | Currently None |
| fg_prct_opp | (not available) | str/null | Currently None |
| 3p_opp | (not available) | int/null | Currently None |
| 3pa_opp | (not available) | int/null | Currently None |
| 3p_prct_opp | (not available) | str/null | Currently None |
| ft_opp | (not available) | int/null | Currently None |
| fta_opp | (not available) | int/null | Currently None |
| ft_prct_opp | (not available) | str/null | Currently None |
| orb_opp | (not available) | int/null | Currently None |
| trb_opp | (not available) | int/null | Currently None |
| ast_opp | (not available) | int/null | Currently None |
| stl_opp | (not available) | int/null | Currently None |
| blk_opp | (not available) | int/null | Currently None |
| tov_opp | (not available) | int/null | Currently None |
| pf_opp | (not available) | int/null | Currently None |

### NBA API Response Structure for Gamelog

The NBA API `teamgamelogs` endpoint returns data in the following structure:

```json
{
  "resultSets": [
    {
      "name": "TeamGameLogs",
      "headers": [
        "SEASON_YEAR", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
        "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN", "FGM", "FGA",
        "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
        "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS"
      ],
      "rowSet": [
        ["2023-24", 1610612738, "BOS", "Boston Celtics", "0022300001",
         "2023-10-25", "BOS vs. NYK", "W", 240, 39, 84, 0.464, 13, 38,
         0.342, 17, 22, 0.773, 11, 37, 48, 26, 7, 5, 14, 20, 108]
      ]
    }
  ]
}
```

### Available but Unused NBA API Columns

These columns are available in the NBA API response but are not currently used in the mapping:

- `SEASON_YEAR`: Already captured in id_season
- `TEAM_ID`: Internal NBA team ID
- `TEAM_NAME`: Full team name (we use abbreviation)
- `GAME_ID`: Unique game identifier
- `MIN`: Minutes played (always 240 for regular games)

These could be added in future versions if needed.

### Columns Not Available in NBA API

The following Basketball Reference columns cannot be directly obtained from the `teamgamelogs` endpoint:

1. **Opponent Statistics**: All opponent stats (fg_opp, pts_opp, etc.)
   - Would require: Separate API call per game using opponent team ID
   - Impact: Significant increase in API calls (82+ additional calls per team)
   - Future Enhancement: Could add optional parameter to fetch opponent stats

2. **pts_opp**: Opponent points
   - Possible workaround: Parse from game result or make additional calls
   - Current status: Set to None

### Data Type Conversions

| Column Type | NBA API Type | BR Type | Conversion |
|-------------|-------------|---------|------------|
| Integers (counts) | int | int | Direct copy |
| Percentages | float (0.485) | str ("0.485") | Format to 3 decimal string |
| Dates | str | str | Direct copy (format may vary) |
| Win/Loss | str ("W"/"L") | str ("W"/"L") | Direct copy |

## Schedule Column Mapping

### Complete Mapping Table

| Basketball Reference Column | NBA API Source Column | Data Type | Transformation Notes |
|----------------------------|----------------------|-----------|---------------------|
| **Metadata** |
| id_season | (added) | int | Season year from input parameter |
| tm | (added) | str | Team abbreviation from input parameter |
| game_date | GAME_DATE | str | Game date |
| time_start | (not available) | str | Set to empty string |
| **Game Info** |
| extdom | MATCHUP (parsed) | str | "@" if away, "" if home |
| opponent | MATCHUP (parsed) | str | Opponent abbreviation |
| w_l | WL | str | "W" for win, "L" for loss |
| overtime | (not available) | str | Set to empty string |
| **Scoring** |
| pts_tm | PTS | int | Team points |
| pts_opp | (not available) | int/null | Currently None |
| **Season Totals** |
| w_tot | WL (calculated) | int | Cumulative wins |
| l_tot | WL (calculated) | int | Cumulative losses |
| streak_w_l | WL (calculated) | str | Format: "W 3" or "L 2" |

### NBA API Response Structure for Schedule

The NBA API `leaguegamefinder` endpoint returns:

```json
{
  "resultSets": [
    {
      "name": "LeagueGameFinderResults",
      "headers": [
        "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
        "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN", "PTS", "FGM",
        "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA",
        "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV",
        "PF", "PLUS_MINUS"
      ],
      "rowSet": [...]
    }
  ]
}
```

### Calculated Columns

These columns are computed from the WL series:

1. **w_tot**: Running count of wins
   ```python
   df['w_tot'] = (df['w_l'] == 'W').cumsum()
   ```

2. **l_tot**: Running count of losses
   ```python
   df['l_tot'] = (df['w_l'] == 'L').cumsum()
   ```

3. **streak_w_l**: Current win/loss streak
   - Format: "W 3" means 3-game win streak
   - Format: "L 2" means 2-game loss streak
   - Calculated by tracking consecutive W/L results

### Columns Not Available in NBA API

1. **time_start**: Game start time
   - Not included in API response
   - Set to empty string
   - Future: Could fetch from different endpoint

2. **overtime**: Overtime indicator
   - Not directly available
   - Could be inferred from regulation time/score
   - Current status: Set to empty string

3. **pts_opp**: Opponent points
   - Not directly available in team-centric query
   - Would require parsing or separate calls
   - Current status: Set to None

## Team Abbreviation Mapping

### Basketball Reference to NBA API

Some team abbreviations differ between the two sources:

| Basketball Reference | NBA API | Notes |
|---------------------|---------|-------|
| BRK | BKN | Brooklyn Nets |
| PHO | PHX | Phoenix Suns |
| CHO | CHA | Charlotte Hornets |

The `WebScrapNBAApi` class handles these differences internally through the `team_mapping` dictionary.

### Team IDs

Each team has a unique NBA API team ID:

```python
{
    'ATL': 1610612737,  # Atlanta Hawks
    'BOS': 1610612738,  # Boston Celtics
    'BRK': 1610612751,  # Brooklyn Nets
    'CHA': 1610612766,  # Charlotte Hornets
    'CHI': 1610612741,  # Chicago Bulls
    'CLE': 1610612739,  # Cleveland Cavaliers
    'DAL': 1610612742,  # Dallas Mavericks
    'DEN': 1610612743,  # Denver Nuggets
    'DET': 1610612765,  # Detroit Pistons
    'GSW': 1610612744,  # Golden State Warriors
    'HOU': 1610612745,  # Houston Rockets
    'IND': 1610612754,  # Indiana Pacers
    'LAC': 1610612746,  # LA Clippers
    'LAL': 1610612747,  # LA Lakers
    'MEM': 1610612763,  # Memphis Grizzlies
    'MIA': 1610612748,  # Miami Heat
    'MIL': 1610612749,  # Milwaukee Bucks
    'MIN': 1610612750,  # Minnesota Timberwolves
    'NOP': 1610612740,  # New Orleans Pelicans
    'NYK': 1610612752,  # New York Knicks
    'OKC': 1610612760,  # Oklahoma City Thunder
    'ORL': 1610612753,  # Orlando Magic
    'PHI': 1610612755,  # Philadelphia 76ers
    'PHO': 1610612756,  # Phoenix Suns (PHX in NBA API)
    'POR': 1610612757,  # Portland Trail Blazers
    'SAC': 1610612758,  # Sacramento Kings
    'SAS': 1610612759,  # San Antonio Spurs
    'TOR': 1610612761,  # Toronto Raptors
    'UTA': 1610612762,  # Utah Jazz
    'WAS': 1610612764   # Washington Wizards
}
```

## Season Format Conversion

NBA API uses a different season format than Basketball Reference:

| Input | Basketball Reference | NBA API |
|-------|---------------------|---------|
| 2024 | 2024 season | "2023-24" |
| 2023 | 2023 season | "2022-23" |
| 2022 | 2022 season | "2021-22" |

Conversion logic:
```python
def _convert_season_format(season: int) -> str:
    return f"{season-1}-{str(season)[-2:]}"
```

## Data Quality Notes

### Percentage Formatting

NBA API returns percentages as decimals (e.g., 0.485), while Basketball Reference displays them as decimal strings (e.g., "0.485"). The scraper converts:

```python
df[col] = df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else None)
```

### Missing Data Handling

When data is unavailable:
- Numeric columns: Set to `None` (becomes `NaN` in pandas)
- String columns: Set to empty string `""`
- Boolean columns: Set to appropriate default

### Date Formats

Date formats may vary in NBA API responses:
- ISO format: "2023-10-25"
- Long format: "Wed, Oct 25, 2023"
- Short format: "10/25/2023"

The scraper passes dates through as-is from the API.

## Future Enhancements

### Planned Improvements

1. **Opponent Statistics Fetching**
   ```python
   # Future method signature
   fetch_nba_api_data(include_opponent_stats: bool = False)
   ```

2. **Advanced Statistics**
   - Offensive Rating (OffRtg)
   - Defensive Rating (DefRtg)
   - Pace
   - True Shooting %

3. **Additional Endpoints**
   - Player game logs
   - Team roster information
   - Advanced box scores

4. **Better Date Handling**
   - Standardize date formats
   - Add timezone support
   - Parse game times when available

### Potential New Columns

Columns that could be added from NBA API:

- `game_id`: NBA's unique game identifier
- `plus_minus`: Team's +/- for the game
- `off_rating`: Offensive rating
- `def_rating`: Defensive rating
- `pace`: Pace of play
- `efg_pct`: Effective field goal %

## References

- [NBA Stats API Documentation](https://stats.nba.com/)
- [Basketball Reference](https://www.basketball-reference.com/)
- [nba_api GitHub](https://github.com/swar/nba_api)

## Version History

- **v1.0** (Current): Basic gamelog and schedule support
  - Core column mapping implemented
  - Opponent stats not included
  - Basic validation and error handling

---

For questions or contributions, please refer to the main project documentation.
