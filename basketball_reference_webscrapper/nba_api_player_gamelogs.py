"""
NBA Boxscore Extractor
Extract player boxscore data for NBA games using nba_api
"""

import pandas as pd
import time
from nba_api.stats.endpoints import boxscoretraditionalv2, leaguegamefinder
from nba_api.stats.static import teams
import json
from datetime import datetime

class NBABoxscoreExtractor:
    """Extract NBA boxscore data with rate limiting"""
    
    def __init__(self, delay=0.6):
        """
        Initialize the extractor
        
        Args:
            delay (float): Delay between API calls in seconds (default 0.6s = ~100 calls/min)
        """
        self.delay = delay
        self.all_teams = teams.get_teams()
        
    def get_team_games(self, team_abbr=None, season='2023-24', season_type='Regular Season'):
        """
        Get all games for a specific team or all teams
        
        Args:
            team_abbr (str): Team abbreviation (e.g., 'LAL', 'BOS'). None for all teams.
            season (str): Season in format 'YYYY-YY' (e.g., '2023-24')
            season_type (str): 'Regular Season', 'Playoffs', 'All Star'
            
        Returns:
            DataFrame with game information
        """
        print(f"Fetching games for season {season}...")
        
        if team_abbr:
            team = [t for t in self.all_teams if t['abbreviation'] == team_abbr][0]
            team_id = team['id']
            
            gamefinder = leaguegamefinder.LeagueGameFinder(
                team_id_nullable=team_id,
                season_nullable=season,
                season_type_nullable=season_type
            )
        else:
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable=season_type
            )
        
        time.sleep(self.delay)
        
        games = gamefinder.get_data_frames()[0]
        print(f"Found {len(games)} game records")
        
        return games
    
    def get_game_boxscore(self, game_id):
        """
        Get player boxscore for a specific game
        
        Args:
            game_id (str): NBA game ID
            
        Returns:
            tuple: (player_stats_df, team_stats_df)
        """
        try:
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            time.sleep(self.delay)
            
            # Get player stats
            player_stats = boxscore.get_data_frames()[0]
            # Get team stats
            team_stats = boxscore.get_data_frames()[1]
            
            return player_stats, team_stats
            
        except Exception as e:
            print(f"Error fetching boxscore for game {game_id}: {e}")
            return None, None
    
    def extract_all_boxscores(self, games_df, max_games=None):
        """
        Extract boxscores for all games in the dataframe
        
        Args:
            games_df (DataFrame): DataFrame with game information (must have GAME_ID column)
            max_games (int): Maximum number of games to process (None for all)
            
        Returns:
            tuple: (all_player_stats_df, all_team_stats_df)
        """
        # Get unique game IDs (since each game appears twice, once for each team)
        unique_game_ids = games_df['GAME_ID'].unique()
        
        if max_games:
            unique_game_ids = unique_game_ids[:max_games]
        
        print(f"\nExtracting boxscores for {len(unique_game_ids)} games...")
        print(f"Estimated time: {(len(unique_game_ids) * self.delay / 60) + (len(unique_game_ids)/10 * 10 / 60) :.1f} minutes")
        
        all_player_stats = []
        all_team_stats = []
        
        for idx, game_id in enumerate(unique_game_ids, 1):
            if idx % 10 == 0:
                print(f"Progress: {idx}/{len(unique_game_ids)} games processed")
                time.sleep(10)  # Extra delay every 10 games
            
            player_stats, team_stats = self.get_game_boxscore(game_id)
            
            if player_stats is not None and team_stats is not None:
                all_player_stats.append(player_stats)
                all_team_stats.append(team_stats)
        
        print(f"\nCompleted! Processed {len(all_player_stats)} games successfully")
        
        # Combine all dataframes
        if all_player_stats:
            combined_player_stats = pd.concat(all_player_stats, ignore_index=True)
            combined_team_stats = pd.concat(all_team_stats, ignore_index=True)
            return combined_player_stats, combined_team_stats
        else:
            return pd.DataFrame(), pd.DataFrame()
    
    def save_to_csv(self, player_stats_df, team_stats_df, prefix='nba_boxscore'):
        """
        Save dataframes to CSV files
        
        Args:
            player_stats_df (DataFrame): Player statistics
            team_stats_df (DataFrame): Team statistics
            prefix (str): Prefix for output files
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        player_file = f"{prefix}_players_{timestamp}.csv"
        team_file = f"{prefix}_teams_{timestamp}.csv"
        
        player_stats_df.to_csv(player_file, index=False)
        team_stats_df.to_csv(team_file, index=False)
        
        print(f"\nData saved:")
        print(f"  - Player stats: {player_file} ({len(player_stats_df)} rows)")
        print(f"  - Team stats: {team_file} ({len(team_stats_df)} rows)")
        
        return player_file, team_file


def main():
    """Example usage"""
    
    # Initialize extractor with 0.6 second delay (safe rate)
    extractor = NBABoxscoreExtractor(delay=0.6)
    
    # Example 1: Get all games for a specific a given season
    all_season_games = extractor.get_team_games(
        season='2025-26',
        season_type='Regular Season'
    )

    print("\n" + "="*60)
    print("Example 2: Extracting boxscores ")
    print("="*60)
    player_stats, team_stats = extractor.extract_all_boxscores(all_season_games)
    
    if not player_stats.empty:
        print(f"\nPlayer stats columns: {list(player_stats.columns)}")
        print(f"\nSample player stats:")
        print(player_stats[['GAME_ID', 'TEAM_ABBREVIATION', 'PLAYER_NAME', 'MIN', 'PTS', 'REB', 'AST']].head(10))
        
        # Save to CSV
        extractor.save_to_csv(player_stats, team_stats)
    
    # # Example 3: Get full season data (commented out - uncomment to run)

    # print("\n" + "="*60)
    # print("Example 3: Full season extraction")
    # print("="*60)
    
    # # For full season, remove max_games limit
    # player_stats_full, team_stats_full = extractor.extract_all_boxscores(lakers_games)

    
    print("\n" + "="*60)
    print("Extraction complete!")
    print("="*60)


if __name__ == "__main__":
    main()