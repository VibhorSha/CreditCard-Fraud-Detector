import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

import yaml
import numpy as np
import pandas as pd

# Configure Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DataSimulationEngine")


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


class FraudSimulator:
    """
    Production-grade synthetic transaction generator mimicking real-world 
    credit card fraud typologies based on the FDH framework.
    """
    def __init__(self, config: dict):
        self.config = config["simulation"]
        self.output_path = Path(config["data"]["raw_dir"]) / config["data"]["raw_file_name"]
        
        # Enforce deterministic generation for pipeline reproducibility
        self.random_state = self.config["random_state"]
        np.random.seed(self.random_state)
        
        # Initialize memory states
        self.customer_profiles = pd.DataFrame()
        self.terminal_profiles = pd.DataFrame()
        
    def generate_profiles(self) -> None:
        """Synthesizes customer and merchant terminal spatial & behavioral profiles."""
        logger.info("Generating customer and terminal spatial profiles...")
        
        # 1. Customer Profiles: ID, location (x,y), mean spending, mean daily frequency
        self.customer_profiles = pd.DataFrame({
            'CUSTOMER_ID': np.arange(self.config["n_customers"]),
            'x_customer_id': np.random.uniform(0, 100, self.config["n_customers"]),
            'y_customer_id': np.random.uniform(0, 100, self.config["n_customers"]),
            'mean_amount': np.random.uniform(5, 100, self.config["n_customers"]),
            'mean_nb_tx_per_day': np.random.uniform(0, 4, self.config["n_customers"])
        })

        # 2. Terminal Profiles: ID, location (x,y)
        self.terminal_profiles = pd.DataFrame({
            'TERMINAL_ID': np.arange(self.config["n_terminals"]),
            'x_terminal_id': np.random.uniform(0, 100, self.config["n_terminals"]),
            'y_terminal_id': np.random.uniform(0, 100, self.config["n_terminals"])
        })

    def assign_terminals_to_customers(self) -> None:
        """Calculates spatial boundaries to map customers to nearby terminals."""
        logger.info(f"Mapping valid terminals within spatial radius {self.config['radius']}...")
        
        def get_nearby_terminals(cust: pd.Series) -> List[int]:
            # Vectorized Euclidean distance calculation
            distances = np.sqrt(
                (self.terminal_profiles['x_terminal_id'] - cust['x_customer_id'])**2 + 
                (self.terminal_profiles['y_terminal_id'] - cust['y_customer_id'])**2
            )
            return self.terminal_profiles.loc[distances < self.config["radius"], 'TERMINAL_ID'].tolist()
            
        self.customer_profiles['available_terminals'] = self.customer_profiles.apply(
            get_nearby_terminals, axis=1
        )

    def generate_transactions(self) -> pd.DataFrame:
        """Executes a Poisson process to generate time-series transactions."""
        logger.info("Generating transactions via exponential time-decay...")
        start_date = datetime.strptime(self.config["start_date"], "%Y-%m-%d")
        transactions = []
        
        for _, cust in self.customer_profiles.iterrows():
            # Calculate total time delta based on frequency
            time_between_tx = np.random.exponential(
                1.0 / cust['mean_nb_tx_per_day'], 
                size=int(self.config["num_days"] * cust['mean_nb_tx_per_day'] * 1.5) # Buffer for variance
            )
            
            # Cumulative sum to get absolute datetime offsets
            tx_times = start_date + pd.to_timedelta(np.cumsum(time_between_tx), unit='d')
            tx_times = tx_times[tx_times < start_date + timedelta(days=self.config["num_days"])]
            
            if len(tx_times) == 0 or len(cust['available_terminals']) == 0:
                continue
                
            cust_txs = pd.DataFrame({
                'TX_DATETIME': tx_times,
                'CUSTOMER_ID': cust['CUSTOMER_ID'],
                'TERMINAL_ID': np.random.choice(cust['available_terminals'], size=len(tx_times)),
                # Transaction amount normally distributed around customer's mean
                'TX_AMOUNT': np.maximum(
                    np.random.normal(cust['mean_amount'], cust['mean_amount'] / 2, size=len(tx_times)), 
                    0.01 # Prevent negative transactions
                )
            })
            transactions.append(cust_txs)
            
        df_tx = pd.concat(transactions, ignore_index=True)
        return df_tx.sort_values('TX_DATETIME').reset_index(drop=True)

    def inject_fraud_scenarios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Labels anomalies based on FDH handbook scenarios."""
        logger.info("Injecting adversarial fraud typologies (Scenarios 1, 2, 3)...")
        df['TX_FRAUD'] = 0
        df['TX_FRAUD_SCENARIO'] = 0
        
        # Scenario 1: Unusually high amounts (Arbitrary heuristic > 220)
        mask_s1 = df['TX_AMOUNT'] > 220
        df.loc[mask_s1, 'TX_FRAUD'] = 1
        df.loc[mask_s1, 'TX_FRAUD_SCENARIO'] = 1
        
        # Scenario 2: Compromised Terminals (2 random terminals per day for 28 days)
        # Scenario 3: Compromised Customers (3 random customers per day for 14 days)
        # Note: Implementing simplified batch vectors for performance
        compromised_terminals = np.random.choice(self.terminal_profiles['TERMINAL_ID'], size=int(self.config["num_days"] * 0.1))
        compromised_customers = np.random.choice(self.customer_profiles['CUSTOMER_ID'], size=int(self.config["num_days"] * 0.1))
        
        mask_s2 = df['TERMINAL_ID'].isin(compromised_terminals) & (np.random.rand(len(df)) < 0.5) # 50% hit rate
        df.loc[mask_s2, 'TX_FRAUD'] = 1
        df.loc[mask_s2, 'TX_FRAUD_SCENARIO'] = 2
        
        mask_s3 = df['CUSTOMER_ID'].isin(compromised_customers) & (np.random.rand(len(df)) < 0.3)
        df.loc[mask_s3, 'TX_AMOUNT'] *= 5 # Fraudsters maximize card limits
        df.loc[mask_s3, 'TX_FRAUD'] = 1
        df.loc[mask_s3, 'TX_FRAUD_SCENARIO'] = 3
        
        # Add sequential Transaction ID
        df.insert(0, 'TRANSACTION_ID', range(len(df)))
        return df

    def execute(self) -> None:
        self.generate_profiles()
        self.assign_terminals_to_customers()
        transactions_df = self.generate_transactions()
        labeled_df = self.inject_fraud_scenarios(transactions_df)
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        labeled_df.to_parquet(self.output_path, engine="pyarrow", compression="snappy", index=False)
        logger.info(f"Dataset Synthesized: {len(labeled_df):,} records saved to {self.output_path}")

if __name__ == "__main__":
    cfg = load_config()
    simulator = FraudSimulator(cfg)
    simulator.execute()