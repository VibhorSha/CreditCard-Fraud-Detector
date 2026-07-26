import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import datetime

# 1. Setup Logging (Observability)
# This tells our program to print messages to the terminal so we know exactly what it's doing.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FraudDataDownloader:
    """Downloads daily fraud data and compiles it into a high-performance Parquet file."""
    
    def __init__(self, base_url: str, raw_dir: str = "data/raw"):
        self.base_url = base_url
        self.raw_dir = Path(raw_dir)
        # Create the data/raw folder if it doesn't exist yet
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_and_compile(self, start_date_str: str, days: int) -> pd.DataFrame:
        """Downloads daily files and stacks them into one DataFrame."""
        start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
        daily_dataframes = []
        
        logger.info(f"Initiating download for {days} days of data starting {start_date}...")
        
        # Loop through each day, download the file, and add it to our list
        for i in tqdm(range(days), desc="Downloading Daily Data"):
            current_date = start_date + datetime.timedelta(days=i)
            url = f"{self.base_url}/{current_date}.pkl"
            
            try:
                # Read the data directly from the web into memory
                df = pd.read_pickle(url)
                daily_dataframes.append(df)
            except Exception as e:
                logger.error(f"Failed to fetch data for {current_date}: {e}")
                
        # Stack all the daily dataframes together
        logger.info("Concatenating daily files...")
        master_df = pd.concat(daily_dataframes, ignore_index=True)
        # Sort them by time so the timeline is perfectly sequential
        master_df = master_df.sort_values('TX_DATETIME').reset_index(drop=True)
        return master_df

if __name__ == "__main__":
    # The URL where the handbook keeps its data
    HANDBOOK_BASE_URL = "https://raw.githubusercontent.com/Fraud-Detection-Handbook/simulated-data-raw/main/data"
    
    downloader = FraudDataDownloader(base_url=HANDBOOK_BASE_URL)
    
    # Download the first 30 days of data
    compiled_data = downloader.fetch_and_compile(start_date_str="2018-04-01", days=30)
    
    # Save it as a Parquet file
    output_path = downloader.raw_dir / "simulated_transactions.parquet"
    logger.info(f"Converting and saving to Parquet format at {output_path}...")
    compiled_data.to_parquet(output_path, engine='pyarrow', index=False)
    
    logger.info(f"Success! Total rows ingested: {len(compiled_data):,}")