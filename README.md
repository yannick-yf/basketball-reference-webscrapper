# WebScrapBasketballReference

WebScrapBasketballReference is a Python package designed to scrape NBA games data from the Basketball Reference website. By providing a URL and a season, the package returns all games' team stats for that season.

## Features

- Scrapes NBA gamelogs, schedules, and player attributes.
- Validates user inputs to ensure data accuracy.
- Handles team-specific data filtering.
- Collects and processes data into a pandas DataFrame.

## Installation

To install WebScrapBasketballReference, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/WebScrapBasketballReference.git
cd WebScrapBasketballReference
pip install -r requirements.txt
```

## Usage

### Importing the Package

```python
from basketball_reference_webscrapper.data_models.feature_model import FeatureIn
from basketball_reference_webscrapper.webscrap_basketball_reference import WebScrapBasketballReference
```

### Creating a FeatureIn Object

```python
feature_object = FeatureIn(
    url='https://www.basketball-reference.com',
    season=2023,
    data_type='gamelog',  # 'gamelog', 'schedule', or 'player_attributes'
    team='all'  # 'all' or a list of team abbreviations e.g., ['BOS', 'LAL']
)
```

### Scraping Data

```python
scraper = WebScrapBasketballReference(feature_object)
data = scraper.webscrappe_nba_games_data()
print(data)
```

## Input Validation

The package performs several input validations:

- **Data Type Validation:** Ensures `data_type` is one of `'gamelog'`, `'schedule'`, or `'player_attributes'`.
- **Season Validation:** Ensures `season` is an integer between 2000 and the current NBA season.
- **Team List Validation:** Ensures `team` is either `'all'` or a list of valid NBA team abbreviations.

## Configuration

The package uses a `params.yaml` file to store URL patterns and other configurations. Ensure this file is correctly set up in the `basketball_reference_webscrapper` directory.

## Logging

Logging is configured to provide information about the scraping execution. Logs can be viewed to monitor the process and debug if necessary.

## Example

```python
from basketball_reference_webscrapper.data_models.feature_model import FeatureIn
from basketball_reference_webscrapper.webscrap_basketball_reference import WebScrapBasketballReference

# Define the feature object
feature_object = FeatureIn(
    url='https://www.basketball-reference.com',
    season=2023,
    data_type='gamelog',
    team='BOS'  # Example team abbreviation for Boston Celtics
)

# Create the scraper instance
scraper = WebScrapBasketballReference(feature_object)

# Scrape the data
data = scraper.webscrappe_nba_games_data()

# Display the data
print(data)
```

## Contributing

Contributions are welcome! Please submit a pull request or create an issue to report bugs or request features.

## License

## Contact

For any questions or feedback, please contact [yannick.flores1992@gmail.com].
