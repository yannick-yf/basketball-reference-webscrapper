"""Test WebScrapNBAApi"""

from unittest import TestCase
from basketball_reference_webscrapper.web_scrap_nba_api import WebScrapNBAApi
from basketball_reference_webscrapper.data_models.feature_model import FeatureIn
import pandas as pd


class TestWebScrapNBAApi(TestCase):
    """Tests for the WebScrapNBAApi class that fetches data from NBA API."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.valid_season = 2023
        self.valid_team = "BOS"
        self.valid_teams_list = ["BOS", "LAL"]

    def test_initialization(self):
        """
        GIVEN valid FeatureIn arguments
        WHEN WebScrapNBAApi is initialized
        THEN it creates an instance successfully
        """
        feature = FeatureIn(data_type="gamelog", season=self.valid_season, team=self.valid_team)
        scraper = WebScrapNBAApi(feature_object=feature)

        assert scraper is not None
        assert scraper.feature_object.data_type == "gamelog"
        assert scraper.feature_object.season == self.valid_season
        assert scraper.feature_object.team == self.valid_team

    def test_team_mapping_exists(self):
        """
        GIVEN a WebScrapNBAApi instance
        WHEN checking team_mapping attribute
        THEN it contains valid NBA team IDs
        """
        feature = FeatureIn(data_type="gamelog", season=self.valid_season, team=self.valid_team)
        scraper = WebScrapNBAApi(feature_object=feature)

        assert scraper.team_mapping is not None
        assert 'BOS' in scraper.team_mapping
        assert scraper.team_mapping['BOS']['team_id'] == 1610612738

    def test_season_format_conversion(self):
        """
        GIVEN a season year
        WHEN converting to NBA API format
        THEN it returns correct format (e.g., 2024 -> "2023-24")
        """
        feature = FeatureIn(data_type="gamelog", season=2024, team=self.valid_team)
        scraper = WebScrapNBAApi(feature_object=feature)

        season_str = scraper._convert_season_format(2024)
        assert season_str == "2023-24"

        season_str = scraper._convert_season_format(2023)
        assert season_str == "2022-23"

    def test_get_nba_team_id(self):
        """
        GIVEN a team abbreviation
        WHEN getting NBA team ID
        THEN it returns correct team ID
        """
        feature = FeatureIn(data_type="gamelog", season=self.valid_season, team=self.valid_team)
        scraper = WebScrapNBAApi(feature_object=feature)

        bos_id = scraper._get_nba_team_id('BOS')
        assert bos_id == 1610612738

        lal_id = scraper._get_nba_team_id('LAL')
        assert lal_id == 1610612747

        # Test invalid team
        invalid_id = scraper._get_nba_team_id('INVALID')
        assert invalid_id is None

    def test_data_type_validation_valid(self):
        """
        GIVEN valid data_type
        WHEN validating
        THEN no error is raised
        """
        feature = FeatureIn(data_type="gamelog", season=self.valid_season, team=self.valid_team)
        scraper = WebScrapNBAApi(feature_object=feature)

        # Should not raise
        scraper._get_data_type_validation()

        feature = FeatureIn(data_type="schedule", season=self.valid_season, team=self.valid_team)
        scraper = WebScrapNBAApi(feature_object=feature)
        scraper._get_data_type_validation()

    def test_data_type_validation_invalid(self):
        """
        GIVEN invalid data_type
        WHEN validating
        THEN ValueError is raised
        """
        with self.assertRaises(ValueError) as context:
            feature = FeatureIn(data_type="invalid_type", season=self.valid_season, team=self.valid_team)
            scraper = WebScrapNBAApi(feature_object=feature)
            scraper._get_data_type_validation()

        assert "not supported" in str(context.exception)

    def test_data_type_validation_none(self):
        """
        GIVEN None data_type
        WHEN validating
        THEN ValueError is raised
        """
        with self.assertRaises(ValueError):
            feature = FeatureIn(data_type=None, season=self.valid_season, team=self.valid_team)
            scraper = WebScrapNBAApi(feature_object=feature)
            scraper._get_data_type_validation()

    def test_season_validation_valid(self):
        """
        GIVEN valid season (>= 2000)
        WHEN validating
        THEN no error is raised
        """
        feature = FeatureIn(data_type="gamelog", season=2023, team=self.valid_team)
        scraper = WebScrapNBAApi(feature_object=feature)
        scraper._get_season_validation()

    def test_season_validation_invalid_year(self):
        """
        GIVEN season before 2000
        WHEN validating
        THEN ValueError is raised
        """
        with self.assertRaises(ValueError) as context:
            feature = FeatureIn(data_type="gamelog", season=1998, team=self.valid_team)
            scraper = WebScrapNBAApi(feature_object=feature)
            scraper._get_season_validation()

        assert "not supported" in str(context.exception)

    def test_season_validation_invalid_type(self):
        """
        GIVEN season as invalid string (non-numeric)
        WHEN creating FeatureIn
        THEN pydantic validation raises error
        """
        # Note: pydantic will coerce valid numeric strings like "2023" to int
        # So we test with a truly invalid string
        with self.assertRaises((ValueError, TypeError)):
            feature = FeatureIn(data_type="gamelog", season="invalid", team=self.valid_team)
            scraper = WebScrapNBAApi(feature_object=feature)
            scraper._get_season_validation()

    def test_team_validation_single_valid(self):
        """
        GIVEN valid single team abbreviation
        WHEN validating
        THEN no error is raised
        """
        feature = FeatureIn(data_type="gamelog", season=self.valid_season, team="BOS")
        scraper = WebScrapNBAApi(feature_object=feature)

        team_list = ['BOS', 'LAL', 'GSW']
        scraper._get_team_list_values_validation(team_list)

    def test_team_validation_list_valid(self):
        """
        GIVEN valid list of team abbreviations
        WHEN validating
        THEN no error is raised
        """
        feature = FeatureIn(data_type="gamelog", season=self.valid_season, team=["BOS", "LAL"])
        scraper = WebScrapNBAApi(feature_object=feature)

        team_list = ['BOS', 'LAL', 'GSW', 'MIA']
        scraper._get_team_list_values_validation(team_list)

    def test_team_validation_all(self):
        """
        GIVEN team="all"
        WHEN validating
        THEN no error is raised
        """
        feature = FeatureIn(data_type="gamelog", season=self.valid_season, team="all")
        scraper = WebScrapNBAApi(feature_object=feature)

        team_list = ['BOS', 'LAL', 'GSW']
        scraper._get_team_list_values_validation(team_list)

    def test_team_validation_invalid_single(self):
        """
        GIVEN invalid team abbreviation
        WHEN validating
        THEN ValueError is raised
        """
        with self.assertRaises(ValueError):
            feature = FeatureIn(data_type="gamelog", season=self.valid_season, team="INVALID")
            scraper = WebScrapNBAApi(feature_object=feature)
            team_list = ['BOS', 'LAL', 'GSW']
            scraper._get_team_list_values_validation(team_list)

    def test_team_validation_invalid_list(self):
        """
        GIVEN list with invalid team abbreviation
        WHEN validating
        THEN ValueError is raised
        """
        with self.assertRaises(ValueError):
            feature = FeatureIn(data_type="gamelog", season=self.valid_season, team=["BOS", "INVALID"])
            scraper = WebScrapNBAApi(feature_object=feature)
            team_list = ['BOS', 'LAL', 'GSW']
            scraper._get_team_list_values_validation(team_list)

    def test_calculate_streak(self):
        """
        GIVEN a series of W/L results
        WHEN calculating streaks
        THEN it returns correct streak strings
        """
        feature = FeatureIn(data_type="schedule", season=self.valid_season, team=self.valid_team)
        scraper = WebScrapNBAApi(feature_object=feature)

        wl_series = pd.Series(['W', 'W', 'W', 'L', 'L', 'W'])
        streaks = scraper._calculate_streak(wl_series)

        expected = pd.Series(['W 1', 'W 2', 'W 3', 'L 1', 'L 2', 'W 1'])
        pd.testing.assert_series_equal(streaks, expected)

    def test_filter_teams_single(self):
        """
        GIVEN single team
        WHEN filtering team reference data
        THEN it returns only that team
        """
        feature = FeatureIn(data_type="gamelog", season=self.valid_season, team="BOS")
        scraper = WebScrapNBAApi(feature_object=feature)

        team_refdata = pd.DataFrame({
            'team_abrev': ['BOS', 'LAL', 'GSW'],
            'city': ['Boston', 'Los Angeles', 'Golden State']
        })

        filtered = scraper._filter_teams(team_refdata)
        assert len(filtered) == 1
        assert filtered['team_abrev'].iloc[0] == 'BOS'

    def test_filter_teams_list(self):
        """
        GIVEN list of teams
        WHEN filtering team reference data
        THEN it returns only those teams
        """
        feature = FeatureIn(data_type="gamelog", season=self.valid_season, team=["BOS", "LAL"])
        scraper = WebScrapNBAApi(feature_object=feature)

        team_refdata = pd.DataFrame({
            'team_abrev': ['BOS', 'LAL', 'GSW', 'MIA'],
            'city': ['Boston', 'Los Angeles', 'Golden State', 'Miami']
        })

        filtered = scraper._filter_teams(team_refdata)
        assert len(filtered) == 2
        assert set(filtered['team_abrev']) == {'BOS', 'LAL'}

    def test_filter_teams_all(self):
        """
        GIVEN team="all"
        WHEN filtering team reference data
        THEN it returns all teams
        """
        feature = FeatureIn(data_type="gamelog", season=self.valid_season, team="all")
        scraper = WebScrapNBAApi(feature_object=feature)

        team_refdata = pd.DataFrame({
            'team_abrev': ['BOS', 'LAL', 'GSW', 'MIA'],
            'city': ['Boston', 'Los Angeles', 'Golden State', 'Miami']
        })

        filtered = scraper._filter_teams(team_refdata)
        assert len(filtered) == 4

    def test_config_loading(self):
        """
        GIVEN a WebScrapNBAApi instance
        WHEN loading config
        THEN it returns valid configuration dictionary
        """
        feature = FeatureIn(data_type="gamelog", season=self.valid_season, team=self.valid_team)
        scraper = WebScrapNBAApi(feature_object=feature)

        config = scraper._load_config()

        assert 'gamelog_nba_api' in config
        assert 'schedule_nba_api' in config
        assert 'list_columns_to_select' in config['gamelog_nba_api']
        assert 'list_columns_to_select' in config['schedule_nba_api']

    def test_team_refdata_loading(self):
        """
        GIVEN a WebScrapNBAApi instance
        WHEN loading team reference data
        THEN it returns valid DataFrame
        """
        feature = FeatureIn(data_type="gamelog", season=self.valid_season, team=self.valid_team)
        scraper = WebScrapNBAApi(feature_object=feature)

        team_refdata = scraper._load_team_refdata()

        assert isinstance(team_refdata, pd.DataFrame)
        assert 'team_abrev' in team_refdata.columns
        assert len(team_refdata) > 0
        assert 'BOS' in team_refdata['team_abrev'].values

    # Note: The following tests would make actual API calls and may fail in cloud environments
    # They are commented out but can be used for local testing

    def test_fetch_nba_api_data_single_team_gamelog(self):
        """
        GIVEN valid FeatureIn for single team gamelog
        WHEN fetch_nba_api_data is called
        THEN it returns DataFrame with expected columns
        """
        feature = FeatureIn(data_type="gamelog", season=2023, team="BOS")
        scraper = WebScrapNBAApi(feature_object=feature)
        df = scraper.fetch_nba_api_data()
    
        assert isinstance(df, pd.DataFrame)
        assert 'id_season' in df.columns
        assert 'tm' in df.columns
        assert 'pts_tm' in df.columns

    def test_fetch_nba_api_data_multiple_teams(self):
        """
        GIVEN valid FeatureIn for multiple teams
        WHEN fetch_nba_api_data is called
        THEN it returns DataFrame with data for all teams
        """
        feature = FeatureIn(data_type="gamelog", season=2023, team=["BOS", "LAL"])
        scraper = WebScrapNBAApi(feature_object=feature)
        df = scraper.fetch_nba_api_data()
    
        assert isinstance(df, pd.DataFrame)
        assert set(df['tm'].unique()) == {'BOS', 'LAL'}

    def test_fetch_nba_api_data_schedule(self):
        """
        GIVEN valid FeatureIn for schedule data
        WHEN fetch_nba_api_data is called
        THEN it returns DataFrame with schedule columns
        """
        feature = FeatureIn(data_type="schedule", season=2023, team="BOS")
        scraper = WebScrapNBAApi(feature_object=feature)
        df = scraper.fetch_nba_api_data()

        assert isinstance(df, pd.DataFrame)
        assert 'w_l' in df.columns
        assert 'w_tot' in df.columns
        assert 'l_tot' in df.columns
        assert 'streak_w_l' in df.columns

    def test_overtime_calculation_no_overtime(self):
        """
        GIVEN a game with 240 total minutes (regulation)
        WHEN overtime is calculated from MIN column
        THEN it returns empty string
        """
        feature = FeatureIn(data_type="schedule", season=2023, team="BOS")
        scraper = WebScrapNBAApi(feature_object=feature)

        # Create test DataFrame with MIN column
        test_df = pd.DataFrame({
            'GAME_DATE': ['2023-10-24'],
            'MATCHUP': ['BOS vs. NYK'],
            'WL': ['W'],
            'PTS': [108],
            'MIN': [240]  # Regular game, no OT
        })

        result_df = scraper._map_schedule_columns(test_df, 'BOS')
        assert result_df['overtime'].iloc[0] == ''

    def test_overtime_calculation_one_overtime(self):
        """
        GIVEN a game with 265 total minutes (1 OT)
        WHEN overtime is calculated from MIN column
        THEN it returns 'OT'
        """
        feature = FeatureIn(data_type="schedule", season=2023, team="BOS")
        scraper = WebScrapNBAApi(feature_object=feature)

        test_df = pd.DataFrame({
            'GAME_DATE': ['2023-10-24'],
            'MATCHUP': ['BOS vs. NYK'],
            'WL': ['W'],
            'PTS': [110],
            'MIN': [265]  # 1 OT
        })

        result_df = scraper._map_schedule_columns(test_df, 'BOS')
        assert result_df['overtime'].iloc[0] == 'OT'

    def test_overtime_calculation_two_overtimes(self):
        """
        GIVEN a game with 290 total minutes (2 OT)
        WHEN overtime is calculated from MIN column
        THEN it returns '2OT'
        """
        feature = FeatureIn(data_type="schedule", season=2023, team="BOS")
        scraper = WebScrapNBAApi(feature_object=feature)

        test_df = pd.DataFrame({
            'GAME_DATE': ['2023-10-24'],
            'MATCHUP': ['BOS vs. NYK'],
            'WL': ['W'],
            'PTS': [115],
            'MIN': [290]  # 2 OT
        })

        result_df = scraper._map_schedule_columns(test_df, 'BOS')
        assert result_df['overtime'].iloc[0] == '2OT'

    def test_overtime_calculation_three_overtimes(self):
        """
        GIVEN a game with 315 total minutes (3 OT)
        WHEN overtime is calculated from MIN column
        THEN it returns '3OT'
        """
        feature = FeatureIn(data_type="schedule", season=2023, team="BOS")
        scraper = WebScrapNBAApi(feature_object=feature)

        test_df = pd.DataFrame({
            'GAME_DATE': ['2023-10-24'],
            'MATCHUP': ['BOS vs. NYK'],
            'WL': ['W'],
            'PTS': [120],
            'MIN': [315]  # 3 OT
        })

        result_df = scraper._map_schedule_columns(test_df, 'BOS')
        assert result_df['overtime'].iloc[0] == '3OT'

    def test_overtime_calculation_via_schedule(self):
        """
        GIVEN OKC Schedule for 2026 season
        THEN the two first games of 2026 should be 2OT
        """
        feature = FeatureIn(data_type="schedule", season=2026, team="OKC")
        scraper = WebScrapNBAApi(feature_object=feature)

        result_df = scraper.fetch_nba_api_data()

        assert result_df['overtime'].iloc[0] == '2OT'
        assert result_df['overtime'].iloc[1] == '2OT'
