import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time

# Observability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FraudFeatureEngineer:
    """Computes high-performance rolling window features for fraud detection."""
    
    def __init__(self, raw_data_path: str, processed_data_dir: str = "data/processed"):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading raw data from {self.raw_data_path}...")
        df = pd.read_parquet(self.raw_data_path)
        # Ensure data is sorted chronologically (Crucial for time-series rolling windows)
        df = df.sort_values('TX_DATETIME').reset_index(drop=True)
        return df

    def get_customer_spending_behavior(self, df: pd.DataFrame, windows_days: list) -> pd.DataFrame:
        # """Calculates Frequency, Sum, and Mean spending features over rolling time windows."""
        logger.info("Calculating customer spending behavior (RFM windows)...")

        df_datetime = df.set_index('TX_DATETIME')

        for window in tqdm(windows_days, desc="Processing Time Windows"):
            # Calculate count, sum, and mean in a single pass
            rolling_stats = df_datetime.groupby('CUSTOMER_ID')['TX_AMOUNT'].rolling(
                f'{window}D', closed='left'
            ).agg(['count', 'sum', 'mean'])

            # Correctly name all three features
            rolling_stats.columns = [
                f'CUSTOMER_ID_NB_TX_{window}DAY_WINDOW',
                f'CUSTOMER_ID_TOT_AMOUNT_{window}DAY_WINDOW',
                f'CUSTOMER_ID_AVG_AMOUNT_{window}DAY_WINDOW'
            ]

            rolling_stats = rolling_stats.reset_index()
            rolling_stats.fillna(0, inplace=True)

            df = pd.merge(df, rolling_stats, on=['TX_DATETIME', 'CUSTOMER_ID'], how='left')

        return df

if __name__ == "__main__":
    RAW_DATA = "data/raw/simulated_transactions.parquet"
    
    engineer = FraudFeatureEngineer(raw_data_path=RAW_DATA)
    
    start_time = time.time()
    
    # 1. Load Data
    transactions_df = engineer.load_data()
    
    # 2. Calculate Customer Behavior (1, 7, and 30 day windows)
    transactions_df = engineer.get_customer_spending_behavior(
        transactions_df, 
        windows_days=[1, 7, 30]
    )
    
    # 3. Save Processed Data
    output_path = engineer.processed_data_dir / "processed_transactions.parquet"
    logger.info(f"Saving engineered features to {output_path}...")
    
    transactions_df.to_parquet(output_path, engine='pyarrow', index=False)
    
    execution_time = time.time() - start_time
    logger.info(f"Feature Engineering Complete in {execution_time:.2f} seconds.")