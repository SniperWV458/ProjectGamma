from curl_cffi import requests
import pandas as pd
import time
import os
import re
import glob
import html
import random
import threading
from datetime import datetime
from typing import List, Tuple, Optional
import calendar
import unicodedata
from urllib.parse import quote
import multiprocessing as mp
from functools import partial


def clean_val(v):
    return "".join(i for i in str(v) if ord(i) < 128)


def _strip_matching_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_dotenv_file() -> None:
    # Load local .env values without overriding already-exported environment vars.
    candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    ]
    seen_paths = set()
    for env_path in candidates:
        env_path = os.path.normpath(env_path)
        if env_path in seen_paths:
            continue
        seen_paths.add(env_path)
        if not os.path.exists(env_path):
            continue

        try:
            # utf-8-sig transparently removes BOM if present (common on Windows).
            with open(env_path, "r", encoding="utf-8-sig") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[len("export ") :].strip()
                    if "=" not in line:
                        continue

                    key, value = line.split("=", 1)
                    key = key.strip().lstrip("\ufeff")
                    if not key or key in os.environ:
                        continue

                    value = _strip_matching_quotes(value.strip())
                    os.environ[key] = value
            return
        except Exception as exc:
            print(f"Warning: failed to read {env_path}: {exc}")


_load_dotenv_file()


def _read_int_env(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
        return max(value, minimum)
    except Exception:
        return default


def _read_float_env(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        value = float(os.getenv(name, str(default)).strip())
        return max(value, minimum)
    except Exception:
        return default


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


class GlobalRequestGate:
    """
    Global cross-thread gate to smooth request bursts.
    - Limits in-flight requests
    - Enforces minimum interval between requests
    - Supports temporary global cooldown after 403/429 bursts
    """

    def __init__(self, max_inflight: int, min_interval_sec: float):
        self._semaphore = threading.BoundedSemaphore(max_inflight)
        self._lock = threading.Lock()
        self._next_allowed_ts = 0.0
        self._cooldown_until_ts = 0.0
        self._min_interval_sec = min_interval_sec

    def acquire(self):
        self._semaphore.acquire()
        while True:
            with self._lock:
                now = time.time()
                target = max(self._next_allowed_ts, self._cooldown_until_ts)
                if now >= target:
                    # Jitter prevents synchronized thread bursts.
                    self._next_allowed_ts = now + self._min_interval_sec + random.uniform(0.02, 0.08)
                    return
                wait_s = target - now
            time.sleep(min(wait_s, 1.0))

    def release(self):
        self._semaphore.release()

    def trigger_cooldown(self, seconds: float):
        with self._lock:
            self._cooldown_until_ts = max(self._cooldown_until_ts, time.time() + max(0.0, seconds))


HTTP_INFLIGHT = _read_int_env("STOCKTWITS_HTTP_INFLIGHT", 3, minimum=1)
HTTP_MIN_INTERVAL = _read_float_env("STOCKTWITS_MIN_INTERVAL", 0.35, minimum=0.0)
HTTP_TIMEOUT = _read_int_env("STOCKTWITS_TIMEOUT_SEC", 20, minimum=5)
STOCKTWITS_IMPERSONATE = os.getenv("STOCKTWITS_IMPERSONATE", "chrome110").strip() or "chrome110"
DEFAULT_USER_AGENT = os.getenv(
    "STOCKTWITS_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
).strip()

REQUEST_GATE = GlobalRequestGate(max_inflight=HTTP_INFLIGHT, min_interval_sec=HTTP_MIN_INTERVAL)


def _build_headers(raw_cookie: str) -> dict:
    headers = {
        "accept": "application/json",
        "accept-language": "en-US,en;q=0.9",
        "referer": "https://stocktwits.com/",
        "user-agent": DEFAULT_USER_AGENT,
    }
    if raw_cookie:
        headers["cookie"] = clean_val(raw_cookie)
    return headers




def _get_cookie_from_env() -> str:
    # Keep cookie out of source code as much as possible.
    return os.getenv("STOCKTWITS_COOKIE", "").strip()


def _safe_get_stream_page(session, url: str, max_id: int, headers: dict):
    REQUEST_GATE.acquire()
    try:
        return session.get(
            url,
            params={"max": int(max_id)},
            headers=headers,
            impersonate=STOCKTWITS_IMPERSONATE,
            timeout=HTTP_TIMEOUT,
        )
    finally:
        REQUEST_GATE.release()


def _load_existing_results(backup_path: str) -> Tuple[pd.DataFrame, set]:
    existing_df = pd.DataFrame()
    existing_months = set()
    if not os.path.exists(backup_path):
        return existing_df, existing_months

    try:
        existing_df = pd.read_csv(backup_path)
    except Exception as e:
        print(f"Error reading backup {backup_path}: {e}")
        return existing_df, existing_months

    if not existing_df.empty and {"Year", "Month"}.issubset(existing_df.columns):
        for _, row in existing_df.iterrows():
            try:
                existing_months.add((int(row["Year"]), int(row["Month"])))
            except Exception:
                pass
    return existing_df, existing_months


def _append_rows_to_backup(rows: List[dict], backup_path: str):
    if not rows:
        return
    frame = pd.DataFrame(rows)
    write_header = not os.path.exists(backup_path) or os.path.getsize(backup_path) == 0
    encoding = "utf-8-sig" if write_header else "utf-8"
    frame.to_csv(
        backup_path,
        mode="a",
        header=write_header,
        index=False,
        encoding=encoding,
    )


def _parse_created_at(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def _normalize_ticker(ticker: str) -> str:
    t = str(ticker).strip().upper()
    if t.startswith("$"):
        t = t[1:]
    t = re.sub(r"\s+", "", t)
    return t


def scrape_historical_multiyear(ticker, date_start, date_end, output_dir):
    """
    Monthly crawl:
    - paginate the StockTwits stream by ID
    - collect all posts whose created_at falls inside target month
    - stop month when oldest item in page is older than month start
    """
    ticker = _normalize_ticker(ticker)
    if not ticker:
        print("Skipping empty ticker.")
        return

    url_ticker = quote(ticker, safe="._-")
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{url_ticker}.json"

    raw_cookie = _get_cookie_from_env()
    if not raw_cookie:
        print(f"[{ticker}] Warning: STOCKTWITS_COOKIE not set; requests may return 403.")
    headers = _build_headers(raw_cookie)

    session = requests.Session()
    results = []
    seen_ids = set()

    s_str, e_str = str(date_start), str(date_end)
    s_year, s_month = int(s_str[:4]), int(s_str[4:])
    e_year, e_month = int(e_str[:4]), int(e_str[4:])

    if e_year >= 2025:
        current_max_id = 620000000
    elif e_year >= 2024:
        current_max_id = 595000000
    elif e_year >= 2023:
        current_max_id = 550000000
    elif e_year >= 2022:
        current_max_id = 480000000
    elif e_year >= 2021:
        current_max_id = 400000000
    elif e_year >= 2020:
        current_max_id = 250000000
    else:
        current_max_id = 160000000

    target_years = range(e_year, s_year - 1, -1)

    backup_path = os.path.join(output_dir, f"{ticker}_History_Backup.csv")
    existing_df, existing_months = _load_existing_results(backup_path)

    if not existing_df.empty:
        print(f"Resuming {ticker} from backup ({len(existing_df)} rows).")
        results = existing_df.to_dict("records")
        if "ID" in existing_df.columns:
            numeric_ids = pd.to_numeric(existing_df["ID"], errors="coerce").dropna()
            seen_ids = set(int(i) for i in numeric_ids.tolist())

    print(
        f"Starting Scrape: {ticker} ({date_start}-{date_end}) "
        f"| month mode | inflight={HTTP_INFLIGHT}, min_interval={HTTP_MIN_INTERVAL:.2f}s"
    )

    last_scraped_date = None
    if existing_months:
        last_scraped_date = min(existing_months, key=lambda x: (x[0], x[1]))

    month_log_every = _read_int_env("STOCKTWITS_MONTH_LOG_EVERY", 40, minimum=5)
    month_max_pages = _read_int_env("STOCKTWITS_MONTH_MAX_PAGES", 20000, minimum=100)
    month_daily_cap = _read_int_env("STOCKTWITS_MAX_POSTS_PER_DAY", 30, minimum=1)
    max_404_retries = _read_int_env("STOCKTWITS_404_RETRIES", 3, minimum=0)
    symbol_not_found = False

    for year in target_years:
        start_m = 12 if year != e_year else e_month
        end_m = 1 if year != s_year else s_month
        target_months = range(start_m, end_m - 1, -1)

        for month in target_months:
            if (year, month) in existing_months and (year, month) != last_scraped_date:
                continue
            if (year, month) == last_scraped_date:
                print(f"Re-scraping last known month {year}-{month:02d} to ensure completeness...")

            days_in_month = calendar.monthrange(year, month)[1]
            month_start_dt = datetime(year, month, 1, 0, 0, 0)
            month_end_dt = datetime(year, month, days_in_month, 23, 59, 59)
            month_rows = []
            day_counts = {}
            pages = 0
            local_cursor = int(current_max_id)
            last_oldest_id = None
            same_cursor_count = 0
            consecutive_403 = 0
            consecutive_429 = 0
            consecutive_404 = 0

            print(
                f"[{ticker}] Scraping month {year}-{month:02d} "
                f"from cursor {local_cursor} (daily_cap={month_daily_cap}) ..."
            )

            while pages < month_max_pages:
                try:

                    r = _safe_get_stream_page(
                        session=session,
                        url=url,
                        max_id=local_cursor,
                        headers=headers,
                    )

                    if r.status_code == 200:
                        consecutive_403 = 0
                        consecutive_429 = 0
                        consecutive_404 = 0
                        msgs = r.json().get("messages", [])
                        if not msgs:
                            local_cursor -= 100000
                            pages += 1
                            time.sleep(random.uniform(0.1, 0.25))
                            continue

                        parsed = []
                        for m in msgs:
                            try:
                                parsed.append((_parse_created_at(m["created_at"]), m))
                            except Exception:
                                continue

                        if not parsed:
                            local_cursor -= 50000
                            pages += 1
                            continue

                        newest_ts = parsed[0][0]
                        oldest_ts = parsed[-1][0]
                        oldest_id = int(parsed[-1][1]["id"])

                        if oldest_id == last_oldest_id:
                            same_cursor_count += 1
                            if same_cursor_count >= 3:
                                local_cursor -= 500000
                                same_cursor_count = 0
                        else:
                            same_cursor_count = 0
                        last_oldest_id = oldest_id

                        # Entire page is newer than month target; keep moving backward.
                        if oldest_ts > month_end_dt:
                            local_cursor = oldest_id - 1
                            pages += 1
                            continue

                        # Entire page is older than month start; month is fully covered.
                        if newest_ts < month_start_dt:
                            local_cursor = oldest_id - 1
                            break

                        in_month_days = {
                            msg_ts.date()
                            for msg_ts, _ in parsed
                            if month_start_dt <= msg_ts <= month_end_dt
                        }
                        if in_month_days and all(day_counts.get(d, 0) >= month_daily_cap for d in in_month_days):
                            local_cursor = oldest_id - 1
                            pages += 1
                            continue

                        for msg_ts, m in parsed:
                            if month_start_dt <= msg_ts <= month_end_dt:
                                day_key = msg_ts.date()
                                if day_counts.get(day_key, 0) >= month_daily_cap:
                                    continue
                                msg_id = int(m["id"])
                                if msg_id in seen_ids:
                                    continue
                                seen_ids.add(msg_id)
                                row = {
                                    "Year": msg_ts.year,
                                    "Month": msg_ts.month,
                                    "Time": m["created_at"],
                                    "ID": msg_id,
                                    "Text": m.get("body", ""),
                                }
                                results.append(row)
                                month_rows.append(row)
                                day_counts[day_key] = day_counts.get(day_key, 0) + 1

                        local_cursor = oldest_id - 1
                        pages += 1

                        if pages % month_log_every == 0:
                            print(
                                f"[{ticker}] {year}-{month:02d} pages={pages}, "
                                f"new_rows={len(month_rows)}, cursor={local_cursor}"
                            )

                        if oldest_ts < month_start_dt:
                            break

                    elif r.status_code == 429:
                        consecutive_429 += 1
                        consecutive_404 = 0
                        backoff = min(90, 5 * (2 ** (consecutive_429 - 1))) + random.uniform(0.2, 1.0)
                        REQUEST_GATE.trigger_cooldown(backoff)
                        print(f"[{ticker}] 429 rate limit. Cooling down for {backoff:.1f}s.")
                        time.sleep(backoff)

                    elif r.status_code == 403:
                        consecutive_403 += 1
                        consecutive_404 = 0
                        backoff = min(180, 8 * (2 ** (consecutive_403 - 1))) + random.uniform(0.5, 1.5)
                        REQUEST_GATE.trigger_cooldown(backoff)
                        print(f"[{ticker}] 403 forbidden. Cooling down for {backoff:.1f}s.")
                        maybe_new_cookie = _get_cookie_from_env()
                        if maybe_new_cookie and maybe_new_cookie != raw_cookie:
                            raw_cookie = maybe_new_cookie
                            headers = _build_headers(raw_cookie)
                            print(f"[{ticker}] Cookie refreshed from environment.")
                        time.sleep(backoff)

                    elif r.status_code == 404:
                        consecutive_404 += 1
                        error_hint = ""
                        try:
                            payload = r.json()
                            error_hint = str(payload.get("response", {}).get("error", "")).strip()
                        except Exception:
                            pass
                        if consecutive_404 <= max_404_retries:
                            retry_wait = min(8.0, 1.5 * (2 ** (consecutive_404 - 1))) + random.uniform(0.2, 0.8)
                            if error_hint:
                                print(
                                    f"[{ticker}] 404 not found ({error_hint}). "
                                    f"Retry {consecutive_404}/{max_404_retries} in {retry_wait:.1f}s."
                                )
                            else:
                                print(
                                    f"[{ticker}] 404 not found. "
                                    f"Retry {consecutive_404}/{max_404_retries} in {retry_wait:.1f}s."
                                )
                            time.sleep(retry_wait)
                            continue

                        if error_hint:
                            print(f"[{ticker}] 404 not found ({error_hint}). Stop symbol.")
                        else:
                            print(f"[{ticker}] 404 not found. Stop symbol.")
                        symbol_not_found = True
                        break

                    else:
                        print(f"[{ticker}] Status {r.status_code}. Retry.")
                        time.sleep(random.uniform(1.0, 2.0))

                except Exception as e:
                    print(f"[{ticker}] Request error: {e}")
                    time.sleep(random.uniform(1.0, 2.5))

                time.sleep(random.uniform(0.08, 0.25))

            if symbol_not_found:
                break

            current_max_id = max(1, int(local_cursor))

            if month_rows:
                _append_rows_to_backup(month_rows, backup_path)
                print(
                    f"[{ticker}] Month {year}-{month:02d} done: pages={pages}, "
                    f"rows={len(month_rows)}, next_cursor={current_max_id}"
                )
            else:
                print(f"[{ticker}] Month {year}-{month:02d} done: no new rows, next_cursor={current_max_id}")

        if symbol_not_found:
            break

    if results:
        final_path = os.path.join(output_dir, f"{ticker}_Full_{date_start}_{date_end}.csv")
        df_final = pd.DataFrame(results)
        if "ID" in df_final.columns:
            df_final = df_final.drop_duplicates(subset=["ID"])
        if "Time" in df_final.columns:
            df_final["Time"] = pd.to_datetime(df_final["Time"], errors="coerce", utc=True)
            df_final = df_final.sort_values("Time")
            df_final["Time"] = df_final["Time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        df_final.to_csv(final_path, index=False, encoding="utf-8-sig")
        print(f"Final Success! File: {final_path}")


def list_news_csv_files(output_dir: str, include_backup: bool = False) -> List[str]:
    pattern = os.path.join(output_dir, "*.csv") if include_backup else os.path.join(output_dir, "*_Full_*.csv")
    files = sorted(glob.glob(pattern))
    
    # Filter out any file that already contains "FinBERT" (case-insensitive)
    filtered = []
    for f in files:
        base = os.path.basename(f)
        # Skip if the filename already includes FinBERT output files
        if "finbert" in base.lower():
            continue
        if include_backup:
            filtered.append(f)
        else:
            if "_History_Backup" not in base:
                filtered.append(f)
    return filtered


def _extract_symbol_from_filename(file_path: str) -> str:
    name = os.path.basename(file_path)
    if "_Full_" in name:
        return name.split("_Full_")[0].upper().strip()
    if "_History_Backup" in name:
        return name.split("_History_Backup")[0].upper().strip()
    return os.path.splitext(name)[0].split("_")[0].upper().strip()


def clean_news_text(raw_text: str) -> str:
    if pd.isna(raw_text):
        return ""

    text = str(raw_text)
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\$(?=[A-Za-z])[A-Za-z]{1,10}\b", " ", text)
    text = re.sub(r"\$", " ", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_news_extracts_from_csv(output_dir: str) -> pd.DataFrame:
    files = list_news_csv_files(output_dir=output_dir, include_backup=False)
    print(f"Loading news extracts from {len(files)} files in: {output_dir}")
    if not files:
        raise FileNotFoundError(f"No non-backup CSV files found in: {output_dir}")

    frames = []
    for file_path in files:
        df = pd.read_csv(file_path)
        if "Text" not in df.columns:
            continue

        frame = df.copy()
        frame["Symbol"] = _extract_symbol_from_filename(file_path)
        frame["SourceFile"] = os.path.basename(file_path)
        frame["RawText"] = frame["Text"].astype(str)
        frame["CleanText"] = frame["RawText"].apply(clean_news_text)
        frame = frame[frame["CleanText"].str.len() > 0].copy()
        frame["Time"] = pd.to_datetime(frame.get("Time"), errors="coerce", utc=True)
        frame["Date"] = frame["Time"].dt.date
        frame = frame[frame["Date"].notna()].copy()
        frames.append(frame)

    if not frames:
        raise ValueError("No valid rows with Text and parseable Time were found.")

    return pd.concat(frames, ignore_index=True)


_FINBERT_BUNDLE = None
_FINBERT_LOCK = threading.Lock()


def _load_finbert(force_reload: bool = False):
    global _FINBERT_BUNDLE
    if _FINBERT_BUNDLE is not None and not force_reload:
        return _FINBERT_BUNDLE

    with _FINBERT_LOCK:
        if _FINBERT_BUNDLE is not None and not force_reload:
            return _FINBERT_BUNDLE

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:
            raise ImportError(
                "FinBERT requires 'torch' and 'transformers'. "
                "Please install them before running sentiment analysis."
            ) from exc

        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        device_pref = os.getenv("FINBERT_DEVICE", "auto").strip().lower()
        if device_pref == "cpu":
            device = torch.device("cpu")
        elif device_pref == "cuda":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        model.eval()
        _FINBERT_BUNDLE = (torch, tokenizer, model, device)
        print(f"FinBERT loaded once on device: {device}")
        return _FINBERT_BUNDLE


def _resolve_label_indices(model) -> Tuple[int, int, int]:
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    positive_idx = next(i for i, label in id2label.items() if "pos" in label)
    negative_idx = next(i for i, label in id2label.items() if "neg" in label)
    neutral_idx = next(i for i, label in id2label.items() if "neu" in label)
    return positive_idx, negative_idx, neutral_idx


def score_texts_with_finbert(
    texts: List[str],
    batch_size: int = 32,
    max_length: int = 256,
    model_bundle: Optional[tuple] = None,
) -> pd.DataFrame:
    if not texts:
        return pd.DataFrame(
            columns=[
                "SentimentLabel",
                "PositiveProb",
                "NegativeProb",
                "NeutralProb",
                "SentimentScore",
            ]
        )

    if model_bundle is None:
        model_bundle = _load_finbert()
    torch, tokenizer, model, device = model_bundle
    positive_idx, negative_idx, neutral_idx = _resolve_label_indices(model)
    rows = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

            for p in probs:
                positive_prob = float(p[positive_idx])
                negative_prob = float(p[negative_idx])
                neutral_prob = float(p[neutral_idx])
                sentiment_score = positive_prob - negative_prob
                if positive_prob >= negative_prob and positive_prob >= neutral_prob:
                    label = "positive"
                elif negative_prob >= positive_prob and negative_prob >= neutral_prob:
                    label = "negative"
                else:
                    label = "neutral"

                rows.append(
                    {
                        "SentimentLabel": label,
                        "PositiveProb": positive_prob,
                        "NegativeProb": negative_prob,
                        "NeutralProb": neutral_prob,
                        "SentimentScore": sentiment_score,
                    }
                )

    return pd.DataFrame(rows)


# def analyze_daily_sentiment_with_finbert(
#     output_dir: str,
#     daily_output_path: str = None,  # Kept for compatibility
#     batch_size: int = 32,
# ) -> None:
#     files = list_news_csv_files(output_dir=output_dir, include_backup=False)
#     print(f"Loading news extracts from {len(files)} files in: {output_dir}")
#     if not files:
#         print(f"No non-backup CSV files found in: {output_dir}")
#         return

#     # Major speed-up: load model/tokenizer once for the whole run.
#     model_bundle = _load_finbert()
#     sentiment_daily_cap = _read_int_env("STOCKTWITS_SENTIMENT_MAX_POSTS_PER_DAY", 30, minimum=0)
#     finbert_max_length = _read_int_env("FINBERT_MAX_LENGTH", 128, minimum=16)
#     save_detailed = _read_bool_env("STOCKTWITS_SAVE_DETAILED", False)

#     print(
#         "Sentiment config: "
#         f"batch_size={batch_size}, max_length={finbert_max_length}, "
#         f"daily_cap={'off' if sentiment_daily_cap <= 0 else sentiment_daily_cap}, "
#         f"save_detailed={save_detailed}"
#     )

#     for file_path in files:
#         symbol = _extract_symbol_from_filename(file_path)
#         print(f"Analyzing sentiment for {symbol} ({os.path.basename(file_path)})...")

#         try:
#             t0 = time.time()
#             df = pd.read_csv(file_path)
#             if "Text" not in df.columns:
#                 print(f"Skipping {file_path}: No 'Text' column.")
#                 continue

#             working_df = df.copy()
#             working_df["CleanText"] = working_df["Text"].astype(str).apply(clean_news_text)
#             working_df = working_df[working_df["CleanText"].str.len() > 0].copy()
#             if working_df.empty:
#                 print(f"Skipping {symbol}: No valid text after cleaning.")
#                 continue

#             working_df["Time"] = pd.to_datetime(working_df.get("Time"), errors="coerce", utc=True)
#             working_df = working_df.dropna(subset=["Time"]).copy()
#             if working_df.empty:
#                 print(f"Skipping {symbol}: No valid timestamps after parsing.")
#                 continue

#             if sentiment_daily_cap > 0:
#                 before_cap = len(working_df)
#                 working_df["Date"] = working_df["Time"].dt.date
#                 # Keep most recent N posts per date to bound inference time on high-volume days.
#                 working_df = (
#                     working_df.sort_values("Time", ascending=False)
#                     .groupby("Date", as_index=False, group_keys=False)
#                     .head(sentiment_daily_cap)
#                     .copy()
#                 )
#                 after_cap = len(working_df)
#                 print(f"[{symbol}] Daily cap applied: {before_cap} -> {after_cap} rows.")
#             else:
#                 working_df["Date"] = working_df["Time"].dt.date

#             filtered_df = working_df.reset_index(drop=True)
#             filtered_texts = filtered_df["CleanText"].tolist()

#             scored_batch = score_texts_with_finbert(
#                 texts=filtered_texts,
#                 batch_size=batch_size,
#                 max_length=finbert_max_length,
#                 model_bundle=model_bundle,
#             )

#             detailed_df = pd.concat(
#                 [filtered_df.reset_index(drop=True), scored_batch.reset_index(drop=True)],
#                 axis=1,
#             )

#             detailed_df = detailed_df.dropna(subset=["Date"])

#             daily_df = (
#                 detailed_df.groupby(["Date"], as_index=False)
#                 .agg(
#                     Posts=("SentimentScore", "size"),
#                     AvgSentiment=("SentimentScore", "mean"),
#                     MedianSentiment=("SentimentScore", "median"),
#                     PositiveProbAvg=("PositiveProb", "mean"),
#                     NegativeProbAvg=("NegativeProb", "mean"),
#                     NeutralProbAvg=("NeutralProb", "mean"),
#                 )
#                 .sort_values(["Date"])
#                 .reset_index(drop=True)
#             )
#             daily_df.insert(0, "Symbol", symbol)

#             base_name = os.path.splitext(os.path.basename(file_path))[0]
#             detailed_save_path = os.path.join(output_dir, f"{base_name}_FinBERT_Detailed.csv")
#             daily_save_path = os.path.join(output_dir, f"{base_name}_FinBERT_Daily.csv")

#             if save_detailed:
#                 detailed_df.to_csv(detailed_save_path, index=False, encoding="utf-8-sig")
#             daily_df.to_csv(daily_save_path, index=False, encoding="utf-8-sig")
#             elapsed = time.time() - t0
#             rate = len(detailed_df) / elapsed if elapsed > 0 else 0.0
#             print(
#                 f"Saved: {daily_save_path} | "
#                 f"rows_scored={len(detailed_df)}, seconds={elapsed:.1f}, rows_per_sec={rate:.1f}"
#             )

#         except Exception as e:
#             print(f"Failed to analyze {symbol}: {e}")

#     return

def analyze_daily_sentiment_with_finbert(
    output_dir: str,
    daily_output_path: str = None,  # Kept for compatibility
    batch_size: int = 32,
) -> None:
    files = list_news_csv_files(output_dir=output_dir, include_backup=False)
    print(f"Loading news extracts from {len(files)} files in: {output_dir}")
    if not files:
        print(f"No non-backup CSV files found in: {output_dir}")
        return

    # Major speed-up: load model/tokenizer once for the whole run.
    model_bundle = _load_finbert()
    sentiment_daily_cap = _read_int_env("STOCKTWITS_SENTIMENT_MAX_POSTS_PER_DAY", 30, minimum=0)
    finbert_max_length = _read_int_env("FINBERT_MAX_LENGTH", 128, minimum=16)
    save_detailed = _read_bool_env("STOCKTWITS_SAVE_DETAILED", False)

    print(
        "Sentiment config: "
        f"batch_size={batch_size}, max_length={finbert_max_length}, "
        f"daily_cap={'off' if sentiment_daily_cap <= 0 else sentiment_daily_cap}, "
        f"save_detailed={save_detailed}"
    )

    for file_path in files:
        symbol = _extract_symbol_from_filename(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]   # e.g. "AAL_Full_201801_202412"
        daily_out = os.path.join(output_dir, f"{base_name}_FinBERT_Daily.csv")
        detailed_out = os.path.join(output_dir, f"{base_name}_FinBERT_Detailed.csv")

        # Skip if sentiment results already exist
        if os.path.exists(daily_out) or os.path.exists(detailed_out):
            print(f"Skipping {symbol}: sentiment output already exists ({daily_out} or {detailed_out})")
            continue

        print(f"Analyzing sentiment for {symbol} ({os.path.basename(file_path)})...")

        try:
            t0 = time.time()
            df = pd.read_csv(file_path)
            if "Text" not in df.columns:
                print(f"Skipping {file_path}: No 'Text' column.")
                continue

            working_df = df.copy()
            working_df["CleanText"] = working_df["Text"].astype(str).apply(clean_news_text)
            working_df = working_df[working_df["CleanText"].str.len() > 0].copy()
            if working_df.empty:
                print(f"Skipping {symbol}: No valid text after cleaning.")
                continue

            working_df["Time"] = pd.to_datetime(working_df.get("Time"), errors="coerce", utc=True)
            working_df = working_df.dropna(subset=["Time"]).copy()
            if working_df.empty:
                print(f"Skipping {symbol}: No valid timestamps after parsing.")
                continue

            if sentiment_daily_cap > 0:
                before_cap = len(working_df)
                working_df["Date"] = working_df["Time"].dt.date
                # Keep most recent N posts per date to bound inference time on high-volume days.
                working_df = (
                    working_df.sort_values("Time", ascending=False)
                    .groupby("Date", as_index=False, group_keys=False)
                    .head(sentiment_daily_cap)
                    .copy()
                )
                after_cap = len(working_df)
                print(f"[{symbol}] Daily cap applied: {before_cap} -> {after_cap} rows.")
            else:
                working_df["Date"] = working_df["Time"].dt.date

            filtered_df = working_df.reset_index(drop=True)
            filtered_texts = filtered_df["CleanText"].tolist()

            scored_batch = score_texts_with_finbert(
                texts=filtered_texts,
                batch_size=batch_size,
                max_length=finbert_max_length,
                model_bundle=model_bundle,
            )

            detailed_df = pd.concat(
                [filtered_df.reset_index(drop=True), scored_batch.reset_index(drop=True)],
                axis=1,
            )

            detailed_df = detailed_df.dropna(subset=["Date"])

            daily_df = (
                detailed_df.groupby(["Date"], as_index=False)
                .agg(
                    Posts=("SentimentScore", "size"),
                    AvgSentiment=("SentimentScore", "mean"),
                    MedianSentiment=("SentimentScore", "median"),
                    PositiveProbAvg=("PositiveProb", "mean"),
                    NegativeProbAvg=("NegativeProb", "mean"),
                    NeutralProbAvg=("NeutralProb", "mean"),
                )
                .sort_values(["Date"])
                .reset_index(drop=True)
            )
            daily_df.insert(0, "Symbol", symbol)

            # Save only the daily file (detailed is optional)
            daily_df.to_csv(daily_out, index=False, encoding="utf-8-sig")
            if save_detailed:
                detailed_df.to_csv(detailed_out, index=False, encoding="utf-8-sig")

            elapsed = time.time() - t0
            rate = len(detailed_df) / elapsed if elapsed > 0 else 0.0
            print(
                f"Saved: {daily_out} | "
                f"rows_scored={len(detailed_df)}, seconds={elapsed:.1f}, rows_per_sec={rate:.1f}"
            )

        except Exception as e:
            print(f"Failed to analyze {symbol}: {e}")

    return


#========================================================= Run in Parallel for Sentiment ==================================================#
# def _process_single_file(file_path, output_dir, batch_size, sentiment_daily_cap, finbert_max_length, save_detailed):
#     """Worker function for a single CSV file."""
#     symbol = _extract_symbol_from_filename(file_path)
#     base_name = os.path.splitext(os.path.basename(file_path))[0]
#     daily_out = os.path.join(output_dir, f"{base_name}_FinBERT_Daily.csv")
#     detailed_out = os.path.join(output_dir, f"{base_name}_FinBERT_Detailed.csv")

#     if os.path.exists(daily_out) or os.path.exists(detailed_out):
#         print(f"Skipping {symbol}: output already exists.")
#         return

#     print(f"Analyzing {symbol}...")
#     try:
#         # Load model inside worker (each worker gets its own copy)
#         model_bundle = _load_finbert()  # this will load once per worker
#         df = pd.read_csv(file_path)
#         if "Text" not in df.columns:
#             return

#         working_df = df.copy()
#         working_df["CleanText"] = working_df["Text"].astype(str).apply(clean_news_text)
#         working_df = working_df[working_df["CleanText"].str.len() > 0].copy()
#         if working_df.empty:
#             return

#         working_df["Time"] = pd.to_datetime(working_df.get("Time"), errors="coerce", utc=True)
#         working_df = working_df.dropna(subset=["Time"]).copy()
#         if working_df.empty:
#             return

#         if sentiment_daily_cap > 0:
#             working_df["Date"] = working_df["Time"].dt.date
#             working_df = (
#                 working_df.sort_values("Time", ascending=False)
#                 .groupby("Date", as_index=False, group_keys=False)
#                 .head(sentiment_daily_cap)
#                 .copy()
#             )
#         else:
#             working_df["Date"] = working_df["Time"].dt.date

#         filtered_df = working_df.reset_index(drop=True)
#         filtered_texts = filtered_df["CleanText"].tolist()

#         scored_batch = score_texts_with_finbert(
#             texts=filtered_texts,
#             batch_size=batch_size,
#             max_length=finbert_max_length,
#             model_bundle=model_bundle,
#         )

#         detailed_df = pd.concat([filtered_df, scored_batch], axis=1)
#         detailed_df = detailed_df.dropna(subset=["Date"])

#         daily_df = (
#             detailed_df.groupby(["Date"], as_index=False)
#             .agg(
#                 Posts=("SentimentScore", "size"),
#                 AvgSentiment=("SentimentScore", "mean"),
#                 MedianSentiment=("SentimentScore", "median"),
#                 PositiveProbAvg=("PositiveProb", "mean"),
#                 NegativeProbAvg=("NegativeProb", "mean"),
#                 NeutralProbAvg=("NeutralProb", "mean"),
#             )
#             .sort_values(["Date"])
#             .reset_index(drop=True)
#         )
#         daily_df.insert(0, "Symbol", symbol)

#         daily_df.to_csv(daily_out, index=False, encoding="utf-8-sig")
#         if save_detailed:
#             detailed_df.to_csv(detailed_out, index=False, encoding="utf-8-sig")

#         print(f"Finished {symbol}: {len(detailed_df)} rows -> {daily_out}")
#     except Exception as e:
#         print(f"Failed {symbol}: {e}")


# def analyze_daily_sentiment_with_finbert_parallel(
#     output_dir: str,
#     batch_size: int = 32,
#     n_workers: int = None,  # if None, use CPU count
# ) -> None:
#     files = list_news_csv_files(output_dir=output_dir, include_backup=False)
#     if not files:
#         print(f"No files found in {output_dir}")
#         return

#     if n_workers is None:
#         n_workers = max(1, mp.cpu_count() - 1)  # leave one core free
#     print(f"Using {n_workers} parallel workers on {len(files)} files")

#     sentiment_daily_cap = _read_int_env("STOCKTWITS_SENTIMENT_MAX_POSTS_PER_DAY", 30)
#     finbert_max_length = _read_int_env("FINBERT_MAX_LENGTH", 128)
#     save_detailed = _read_bool_env("STOCKTWITS_SAVE_DETAILED", False)

#     # Create a partial function with fixed arguments
#     worker_func = partial(
#         _process_single_file,
#         output_dir=output_dir,
#         batch_size=batch_size,
#         sentiment_daily_cap=sentiment_daily_cap,
#         finbert_max_length=finbert_max_length,
#         save_detailed=save_detailed,
#     )

#     with mp.Pool(processes=n_workers) as pool:
#         pool.map(worker_func, files)