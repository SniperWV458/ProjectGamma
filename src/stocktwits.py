from curl_cffi import requests
import pandas as pd
import time
import os
from datetime import datetime

def clean_val(v):
    return "".join(i for i in str(v) if ord(i) < 128)

def scrape_historical_multiyear(ticker):
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    
    # --- STEP 1: Paste your cookie here ---
    raw_cookie = "YOUR_COOKIE_HERE" 
    
    headers = {
        "accept": "application/json",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "cookie": clean_val(raw_cookie)
    }

    results = []
    # Starting slightly higher to ensure 2024-12 is captured
    # current_max_id = 598000000
    current_max_id = 152207798
    
    target_years = range(2018, 2011, -1)
    target_months = range(12, 0, -1)

    print(f"Starting Mega-Scrape: {ticker} (2009-2024) - Timeout Enhanced")

    for year in target_years:
        for month in target_months:
            found_month = False
            attempts = 0
            last_seen_id = None
            stuck_count = 0
            
            while not found_month and attempts < 15:
                try:
                    # Increased timeout to 30 seconds to avoid curl(28)
                    print(f"Requesting using url and current_max_id: {url}?%20&max={current_max_id}")
                    r = requests.get(url, params={"max": current_max_id}, headers=headers, impersonate="chrome110", timeout=30)
                    
                    if r.status_code == 200:
                        msgs = r.json().get('messages', [])
                        if not msgs: 
                            current_max_id += 2000000
                            attempts += 1
                            continue
                        
                        top_id = msgs[0]['id']
                        if top_id == last_seen_id:
                            stuck_count += 1
                        else:
                            stuck_count = 0
                        last_seen_id = top_id

                        # When we keep getting the same page, force a bigger backward jump
                        if stuck_count >= 4:
                            current_max_id -= 2500000
                            print(f"Detected oscillation at id {top_id}, forcing jump to {current_max_id}")
                            attempts += 1
                            time.sleep(1.5)
                            continue
                        
                        ts = datetime.strptime(msgs[0]['created_at'], '%Y-%m-%dT%H:%M:%SZ')
                        
                        if ts.year > year or (ts.year == year and ts.month > month):
                            m_diff = (ts.year - year) * 12 + (ts.month - month)
                            jump = 3400000 * m_diff if ts.year > 2020 else 1000000 * m_diff
                            current_max_id -= max(jump, 100000)
                        elif ts.year < year or (ts.year == year and ts.month < month):
                            current_max_id += 500000
                        else:
                            month_data = msgs
                            temp_id = msgs[-1]['id']
                            for _ in range(3):
                                # Sub-requests also use 30s timeout
                                print(f"Requesting using url and temp_id: {url}?%20&max={temp_id}")
                                nr = requests.get(url, params={"max": temp_id}, headers=headers, impersonate="chrome110", timeout=30)
                                if nr.status_code == 200:
                                    n_msgs = nr.json().get('messages', [])
                                    if not n_msgs: break
                                    month_data.extend(n_msgs)
                                    temp_id = n_msgs[-1]['id']
                                    if len(month_data) >= 100: break
                                time.sleep(1.5)

                            for m in month_data[:100]:
                                results.append({
                                    'Year': year, 'Month': month,
                                    'Time': m['created_at'], 'ID': m['id'],
                                    'Text': m['body']
                                })
                            
                            print(f"DONE: [{year}-{month:02d}] -> Found: {msgs[0]['created_at']} (Total: {len(results)})")
                            current_max_id = month_data[-1]['id'] - 1500000
                            found_month = True
                    elif r.status_code == 403:
                        print("ERROR 403: Check Cookie.")
                        return
                    else:
                        print(f"HTTP {r.status_code} at {year}-{month}, retrying...")
                        time.sleep(5)
                except Exception as e:
                    # If timeout happens, wait 5 seconds and stay in the same loop to retry
                    print(f"Connection Issue: {e}. Retrying in 5s...")
                    time.sleep(5)
                
                attempts += 1
                time.sleep(1.5)

            # Auto-backup
            if results and found_month:
                df_b = pd.DataFrame(results)
                b_path = os.path.join(os.path.expanduser("~"), "Desktop", f"{ticker}_History_Backup.csv")
                df_b.to_csv(b_path, index=False, encoding='utf-8-sig')

    if results:
        final_path = os.path.join(os.path.expanduser("~"), "Desktop", f"{ticker}_Full_2009_2024.csv")
        pd.DataFrame(results).to_csv(final_path, index=False, encoding='utf-8-sig')
        print(f"Final Success! File: {final_path}")

scrape_historical_multiyear("USO")