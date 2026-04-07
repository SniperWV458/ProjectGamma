import pandas as pd
import concurrent.futures
from stocktwits_analysis import (
    scrape_historical_multiyear,
    analyze_daily_sentiment_with_finbert
)
import os
import time
import random
import re




def _normalize_symbol(symbol) -> str:
    s = str(symbol).strip().upper()
    if s.startswith("$"):
        s = s[1:]
    s = re.sub(r"\s+", "", s)
    return s


def process_single_symbol(symbol, date_start, date_end, output_dir):
    """Helper function to scrape a single symbol safely."""
    # Small startup jitter prevents all workers from firing at the same instant.
    start_jitter_max = float(os.getenv("STOCKTWITS_THREAD_START_JITTER", "1.0"))
    if start_jitter_max > 0:
        time.sleep(random.uniform(0.0, start_jitter_max))
    symbol_norm = _normalize_symbol(symbol)
    if not symbol_norm:
        print(f"Skipping invalid empty symbol: {symbol}")
        return None
    print(f"Starting thread for: {symbol_norm}")
    try:
        scrape_historical_multiyear(symbol_norm, date_start, date_end, output_dir)
        return symbol_norm
    except Exception as e:
        print(f"Failed to process {symbol_norm}: {e}")
        return None


def run_scraper(symbols_file, date_start, date_end, output_dir, max_workers=4):
    processed_symbols = []
    try:
        if symbols_file.endswith(".csv"):
             df = pd.read_csv(symbols_file)
             symbols_raw = df["ticker"].dropna().unique().tolist()
        else: # Handle list directly if passed
             symbols_raw = symbols_file

        symbols = []
        seen = set()
        for s in symbols_raw:
            s_clean = _normalize_symbol(s)
            if not s_clean or s_clean in seen:
                continue
            seen.add(s_clean)
            symbols.append(s_clean)
        
        # --- Group Symbols by Status ---
        groups = {'a': [], 'b': [], 'c': []}
        
        print(f"Classifying {len(symbols)} symbols...")
        for s in symbols:
            s_clean = s
            full_file = os.path.join(output_dir, f"{s_clean}_Full_{date_start}_{date_end}.csv")
            backup_file = os.path.join(output_dir, f"{s_clean}_History_Backup.csv")
            
            if os.path.exists(full_file):
                groups['c'].append(s_clean)
            elif os.path.exists(backup_file):
                groups['b'].append(s_clean)
            else:
                groups['a'].append(s_clean)

        print("=" * 60)
        print(f"Status Summary:")
        print(f"  (a) Not Started [No files]         : {len(groups['a'])}")
        print(f"  (b) Incomplete  [Backup only]      : {len(groups['b'])}")
        print(f"  (c) Completed   [Full file exists] : {len(groups['c'])}")
        print("=" * 60)
        
        symbols_to_process = []
        
        # Priority 1: Situation (b) - Incomplete -> Resume
        if groups['b']:
            print(f"1. Adding {len(groups['b'])} incomplete symbols (Group b) to queue...")
            symbols_to_process.extend(groups['b'])
            
        # Priority 2: Situation (a) - New -> Start
        if groups['a']:
            print(f"2. Adding {len(groups['a'])} new symbols (Group a) to queue...")
            symbols_to_process.extend(groups['a'])
            
        # # Priority 1: Situation (a) - New -> Start
        # if groups['a']:
        #     print(f"1. Adding {len(groups['a'])} new symbols (Group a) to queue...")
        #     symbols_to_process.extend(groups['a'])
        
        # # Priority 2: Situation (b) - Incomplete -> Resume
        # if groups['b']:
        #     print(f"2. Adding {len(groups['b'])} incomplete symbols (Group b) to queue...")
        #     symbols_to_process.extend(groups['b'])

            
        # Situation (c) - Completed -> SKIPPED
        if groups['c']:
            print(f"3. Skipping {len(groups['c'])} completed symbols (Group c).")

        print(f"Total symbols queued: {len(symbols_to_process)}")
        print(f"Using {max_workers} worker threads.")
        print("=" * 60)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(process_single_symbol, symbol, date_start, date_end, output_dir): symbol
                for symbol in symbols_to_process
            }

            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        processed_symbols.append(result)
                        print(f"Finished processing: {result}")
                except Exception as exc:
                    print(f"{symbol} generated an exception: {exc}")

    except Exception as e:
        print(f"An error occurred: {e}")
    return processed_symbols


def run_finbert_daily_sentiment(output_dir, batch_size=32):
    # No longer returning a single giant DataFrame
    analyze_daily_sentiment_with_finbert(
        output_dir=output_dir,
        batch_size=batch_size,
    )
    print("Sentiment analysis completed file-by-file.")
    return None

if __name__ == "__main__":

    # Configure pipeline
    SYMBOLS_FILE = "linked_secids_top_liquid_2018_2024_sample_100pct.csv"
    DATE_START = 201801
    DATE_END = 202412
    # OUTPUT_DIR = "D:\\stocktwits_output_daily"
    # OUTPUT_DIR = r"C:\Users\Dell\Desktop\HKU MFFinTech\capstone\ProjectGamma-main\output"
    OUTPUT_DIR = r"D:\桌面\stocktwist0404\stocktwits_output_daily"
    BATCH_SIZE = 32
    MAX_WORKERS = 8

    RUN_SCRAPING = True
    RUN_FINBERT_SENTIMENT = False


    if RUN_SCRAPING:
         run_scraper(SYMBOLS_FILE, DATE_START, DATE_END, OUTPUT_DIR, max_workers=MAX_WORKERS)
    # if RUN_FINBERT_SENTIMENT:
    #     run_finbert_daily_sentiment(OUTPUT_DIR, batch_size=BATCH_SIZE)
    #     # analyze_daily_sentiment_with_finbert_parallel(OUTPUT_DIR, batch_size=BATCH_SIZE, n_workers=10)
