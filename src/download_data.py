import logging
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import datetime

# 1. Total Observability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FraudDataDownloader:
    """Production-grade data ingestion with defensive streaming and idempotency."""
    
    def __init__(self, base_url: str, raw_dir: str = "data/raw"):
        self.base_url = base_url
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a temporary directory for safe checkpointing
        self.temp_dir = self.raw_dir / "temp_pkls"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file_safely(self, url: str, dest_path: Path):
        """Streams a file with byte-level observability and strict timeouts."""
        if dest_path.exists():
            return  # Idempotent design: Skip if we already downloaded it on a previous run
            
        # Defensive Engineering: 30-second timeout
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status() 
        
        total_size = int(response.headers.get('content-length', 0))
        
        # Byte-level progress bar
        with open(dest_path, 'wb') as file, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            leave=False # Hides the bar when this specific file finishes
        ) as bar:
            # Download in 8KB chunks so we don't overwhelm memory
            for chunk in response.iter_content(chunk_size=8192):
                size = file.write(chunk)
                bar.update(size)

    def fetch_and_compile(self, start_date_str: str, days: int) -> pd.DataFrame:
        start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
        daily_dataframes = []
        
        logger.info(f"Initiating secure download for {days} days...")
        
        # Master progress bar for the overall job
        for i in tqdm(range(days), desc="Overall Progress"):
            current_date = start_date + datetime.timedelta(days=i)
            url = f"{self.base_url}/{current_date}.pkl"
            dest_path = self.temp_dir / f"{current_date}.pkl"
            
            try:
                # Step 1: Download securely to local disk
                self.download_file_safely(url, dest_path)
                
                # Step 2: Read from local disk into Pandas
                df = pd.read_pickle(dest_path)
                daily_dataframes.append(df)
            except Exception as e:
                logger.error(f"Failed processing {current_date}: {e}")
                
        logger.info("Concatenating daily files...")
        master_df = pd.concat(daily_dataframes, ignore_index=True)
        master_df = master_df.sort_values('TX_DATETIME').reset_index(drop=True)
        return master_df

if __name__ == "__main__":
    HANDBOOK_BASE_URL = "https://raw.githubusercontent.com/Fraud-Detection-Handbook/simulated-data-raw/main/data"
    
    downloader = FraudDataDownloader(base_url=HANDBOOK_BASE_URL)
    
    # Let's pull the data
    compiled_data = downloader.fetch_and_compile(start_date_str="2018-04-01", days=150)
    
    output_path = downloader.raw_dir / "simulated_transactions.parquet"
    logger.info(f"Converting and saving to Parquet format at {output_path}...")
    
    # Save as Parquet
    compiled_data.to_parquet(output_path, engine='pyarrow', index=False)
    
    logger.info(f"Success! Total rows ingested: {len(compiled_data):,}")