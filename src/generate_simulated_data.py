import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


def generate_customer_profiles_table(n_customers: int, random_state: int = 0) -> pd.DataFrame:
    """Generates customer profiles with spatial and spending characteristics."""
    np.random.seed(random_state)
    customer_id_properties = []
    
    for customer_id in range(n_customers):
        x_customer_id = np.random.uniform(0, 100)
        y_customer_id = np.random.uniform(0, 100)
        mean_amount = np.random.uniform(5, 100)
        std_amount = mean_amount / 2
        mean_nb_tx_per_day = np.random.uniform(0, 4)
        
        customer_id_properties.append([
            customer_id, x_customer_id, y_customer_id, 
            mean_amount, std_amount, mean_nb_tx_per_day
        ])
        
    return pd.DataFrame(
        customer_id_properties, 
        columns=['CUSTOMER_ID', 'x_customer_id', 'y_customer_id', 'mean_amount', 'std_amount', 'mean_nb_tx_per_day']
    )


def generate_terminal_profiles_table(n_terminals: int, random_state: int = 1) -> pd.DataFrame:
    """Generates terminal (merchant POS) profiles with spatial coordinates."""
    np.random.seed(random_state)
    terminal_id_properties = []
    
    for terminal_id in range(n_terminals):
        x_terminal_id = np.random.uniform(0, 100)
        y_terminal_id = np.random.uniform(0, 100)
        terminal_id_properties.append([terminal_id, x_terminal_id, y_terminal_id])
        
    return pd.DataFrame(
        terminal_id_properties, 
        columns=['TERMINAL_ID', 'x_terminal_id', 'y_terminal_id']
    )


def get_list_terminals_within_radius(customer_profile: pd.Series, x_y_terminals: np.ndarray, r: float) -> list[int]:
    """Finds all terminal IDs within distance r of a given customer's location."""
    x_y_customer = customer_profile[['x_customer_id', 'y_customer_id']].values.astype(float)
    squared_diff_x_y = np.square(x_y_customer - x_y_terminals)
    dist_x_y = np.sqrt(np.sum(squared_diff_x_y, axis=1))
    return list(np.where(dist_x_y < r)[0])


def generate_transactions_table(customer_profile: pd.Series, start_date: str = "2018-04-01", nb_days: int = 90) -> pd.DataFrame:
    """Simulates transactions for a single customer over nb_days."""
    customer_transactions = []
    random.seed(int(customer_profile.CUSTOMER_ID))
    np.random.seed(int(customer_profile.CUSTOMER_ID))
    
    for day in range(nb_days):
        nb_tx = np.random.poisson(customer_profile.mean_nb_tx_per_day)
        if nb_tx > 0:
            for tx in range(nb_tx):
                time_tx = int(np.random.normal(86400 / 2, 20000))
                if 0 < time_tx < 86400:
                    amount = np.random.normal(customer_profile.mean_amount, customer_profile.std_amount)
                    if amount < 0:
                        amount = np.random.uniform(0, customer_profile.mean_amount * 2)
                    amount = np.round(amount, decimals=2)
                    
                    if len(customer_profile.available_terminals) > 0:
                        terminal_id = random.choice(customer_profile.available_terminals)
                        customer_transactions.append([
                            time_tx + day * 86400, day, customer_profile.CUSTOMER_ID, terminal_id, amount
                        ])
                        
    customer_transactions = pd.DataFrame(
        customer_transactions, 
        columns=['TX_TIME_SECONDS', 'TX_TIME_DAYS', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT']
    )
    
    if len(customer_transactions) > 0:
        customer_transactions['TX_DATETIME'] = pd.to_datetime(
            customer_transactions["TX_TIME_SECONDS"], unit='s', origin=start_date
        )
        customer_transactions = customer_transactions[[
            'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS'
        ]]
        
    return customer_transactions


def add_frauds(customer_profiles_table: pd.DataFrame, terminal_profiles_table: pd.DataFrame, transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Injects 3 fraud scenarios according to the Handbook specifications."""
    transactions_df = transactions_df.copy()
    transactions_df['TX_FRAUD'] = 0
    transactions_df['TX_FRAUD_SCENARIO'] = 0
    
    # Scenario 1: High amount fraud (> 220)
    s1_mask = transactions_df.TX_AMOUNT > 220
    transactions_df.loc[s1_mask, 'TX_FRAUD'] = 1
    transactions_df.loc[s1_mask, 'TX_FRAUD_SCENARIO'] = 1
    
    # Scenario 2: Compromised Terminals (2 per day for 28 days)
    for day in range(transactions_df.TX_TIME_DAYS.max() + 1):
        compromised_terminals = terminal_profiles_table.TERMINAL_ID.sample(n=2, random_state=day)
        compromised_transactions = transactions_df[
            (transactions_df.TX_TIME_DAYS >= day) & 
            (transactions_df.TX_TIME_DAYS < day + 28) & 
            (transactions_df.TERMINAL_ID.isin(compromised_terminals))
        ]
        transactions_df.loc[compromised_transactions.index, 'TX_FRAUD'] = 1
        transactions_df.loc[compromised_transactions.index, 'TX_FRAUD_SCENARIO'] = 2

    # Scenario 3: Compromised Cards / Skimming (3 per day for 14 days, 1/3 transactions multiplied 5x)
    for day in range(transactions_df.TX_TIME_DAYS.max() + 1):
        compromised_customers = customer_profiles_table.CUSTOMER_ID.sample(n=3, random_state=day).values
        compromised_transactions = transactions_df[
            (transactions_df.TX_TIME_DAYS >= day) & 
            (transactions_df.TX_TIME_DAYS < day + 14) & 
            (transactions_df.CUSTOMER_ID.isin(compromised_customers))
        ]
        nb_compromised_transactions = len(compromised_transactions)
        random.seed(day)
        if nb_compromised_transactions > 0:
            index_frauds = random.sample(
                list(compromised_transactions.index.values), 
                k=int(nb_compromised_transactions / 3)
            )
            transactions_df.loc[index_frauds, 'TX_AMOUNT'] = transactions_df.loc[index_frauds, 'TX_AMOUNT'] * 5
            transactions_df.loc[index_frauds, 'TX_FRAUD'] = 1
            transactions_df.loc[index_frauds, 'TX_FRAUD_SCENARIO'] = 3
            
    return transactions_df


def generate_and_save_dataset(
    n_customers: int = 5000, 
    n_terminals: int = 10000, 
    nb_days: int = 90, 
    start_date: str = "2018-04-01", 
    r: float = 5.0,
    output_dir: str = "data/raw"
) -> None:
    """Executes full generation pipeline and saves daily Parquet partitions."""
    print("🚀 Starting Data Simulation Engine...")
    start_time = time.time()
    
    # 1. Profiles
    print("👤 Generating Customer Profiles...")
    customer_profiles = generate_customer_profiles_table(n_customers, random_state=0)
    
    print("🏪 Generating Terminal Profiles...")
    terminal_profiles = generate_terminal_profiles_table(n_terminals, random_state=1)
    
    print("📍 Associating Terminals to Customers...")
    x_y_terminals = terminal_profiles[['x_terminal_id', 'y_terminal_id']].values.astype(float)
    customer_profiles['available_terminals'] = customer_profiles.apply(
        lambda x: get_list_terminals_within_radius(x, x_y_terminals=x_y_terminals, r=r), axis=1
    )
    
    # 2. Transactions
    print(f"💳 Simulating Transactions for {nb_days} days...")
    tx_list = []
    for _, customer_row in tqdm(customer_profiles.iterrows(), total=len(customer_profiles), desc="Customers"):
        tx_df = generate_transactions_table(customer_row, start_date=start_date, nb_days=nb_days)
        if len(tx_df) > 0:
            tx_list.append(tx_df)
            
    transactions_df = pd.concat(tx_list, ignore_index=True)
    
    # Sort chronologically & assign unique TRANSACTION_ID
    transactions_df = transactions_df.sort_values('TX_DATETIME').reset_index(drop=True)
    transactions_df['TRANSACTION_ID'] = transactions_df.index
    
    # Reorder columns
    transactions_df = transactions_df[[
        'TRANSACTION_ID', 'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 
        'TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS'
    ]]
    
    # 3. Add Fraud Scenarios
    print("😈 Injecting Fraud Scenarios...")
    transactions_df = add_frauds(customer_profiles, terminal_profiles, transactions_df)
    
    # 4. Save to Parquet partitioned by Day
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving partitioned Apache Parquet files to '{output_dir}/'...")
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    
    for day in range(nb_days):
        day_df = transactions_df[transactions_df.TX_TIME_DAYS == day].copy()
        current_date_str = (start_datetime + timedelta(days=day)).strftime("%Y-%m-%d")
        file_filename = out_path / f"{current_date_str}.parquet"
        
        # Optimize types for disk storage
        day_df['TRANSACTION_ID'] = day_df['TRANSACTION_ID'].astype('uint64')
        day_df['CUSTOMER_ID'] = day_df['CUSTOMER_ID'].astype('uint32')
        day_df['TERMINAL_ID'] = day_df['TERMINAL_ID'].astype('uint32')
        day_df['TX_AMOUNT'] = day_df['TX_AMOUNT'].astype('float32')
        day_df['TX_FRAUD'] = day_df['TX_FRAUD'].astype('uint8')
        day_df['TX_FRAUD_SCENARIO'] = day_df['TX_FRAUD_SCENARIO'].astype('uint8')
        
        day_df.to_parquet(file_filename, index=False, engine='pyarrow')
        
    total_tx = len(transactions_df)
    total_frauds = transactions_df.TX_FRAUD.sum()
    fraud_pct = (total_frauds / total_tx) * 100
    
    print("\n" + "=" * 50)
    print("✅ DATASET GENERATION COMPLETE")
    print(f"⏱️ Total Execution Time: {time.time() - start_time:.2f} seconds")
    print(f"📊 Total Transactions: {total_tx:,}")
    print(f"🚨 Total Fraudulent Txs: {total_frauds:,} ({fraud_pct:.2f}%)")
    print("=" * 50)


if __name__ == "__main__":
    generate_and_save_dataset()