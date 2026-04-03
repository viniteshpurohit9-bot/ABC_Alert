"""
Candle A-B-C-DExit-DEntry Structure (Sell Setup) — TEST MODE
=============================================================
Scans top 10 Nifty stocks on 1-hour Heikin Ashi candles.
Prints an alert whenever the last close is within 10 points of dEntryLine.

Install dependencies:
    pip install pandas numpy yfinance

Run:
    python abc_dentry_test.py
"""

import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
NIFTY_10 = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "KOTAKBANK.NS",
]

INTERVAL       = "1h"    # 1-hour candles
PERIOD         = "30d"   # Last 30 days (yfinance supports up to 730d for 1h)
ALERT_DISTANCE = 10      # Points near dEntryLine to trigger alert
MIN_TICK       = 0.05    # Minimum tick size for NSE stocks

# Telegram Settings (Get these from @BotFather and @userinfobot)
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID   = ""


# ─────────────────────────────────────────
#  HEIKIN ASHI
# ─────────────────────────────────────────
def compute_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = df.copy()
    ha["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

    ha_open = np.zeros(len(df))
    ha_open[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha["ha_close"].iloc[i - 1]) / 2
    ha["ha_open"] = ha_open

    ha["ha_high"] = ha[["high", "ha_open", "ha_close"]].max(axis=1)
    ha["ha_low"]  = ha[["low",  "ha_open", "ha_close"]].min(axis=1)

    ha["open"]  = ha["ha_open"]
    ha["high"]  = ha["ha_high"]
    ha["low"]   = ha["ha_low"]
    ha["close"] = ha["ha_close"]

    return ha[["open", "high", "low", "close", "volume"]].reset_index(drop=True)


# ─────────────────────────────────────────
#  STRUCTURE DETECTION
# ─────────────────────────────────────────
def detect_structure(df: pd.DataFrame) -> dict:
    result = {"found": False, "dEntryLine": None}

    wait_for_b = wait_for_c = wait_for_dexit = wait_for_dentry = False
    highest_wick = highest_wick_d = np.nan
    a_high = b_low = c_high = dexit_low = np.nan
    a_idx  = b_idx = c_idx  = dexit_idx = -1

    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values

    for i in range(1, len(df)):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]

        prev_body_high = max(opens[i - 1], closes[i - 1])
        prev_body_low  = min(opens[i - 1], closes[i - 1])
        curr_body_high = max(o, c)
        curr_body_low  = min(o, c)

        # ── Candle A: bearish inside bar ──
        inside_body = (curr_body_high <= prev_body_high) and (curr_body_low >= prev_body_low)
        candle_a    = inside_body and (o > c)

        if candle_a:
            wait_for_b      = True
            wait_for_c      = False
            wait_for_dexit  = False
            wait_for_dentry = False
            highest_wick    = h
            a_high = h
            a_idx  = i
            continue

        if wait_for_b:
            highest_wick = max(highest_wick, h)

        # ── Candle B: close breaks above previous bar's high ──
        candle_b = wait_for_b and (c > highs[i - 1])
        if candle_b:
            wait_for_b = False
            wait_for_c = True
            b_low = l
            b_idx = i
            continue

        # ── Candle C: bearish candle ──
        candle_c = wait_for_c and (c < o)
        if candle_c:
            wait_for_c     = False
            wait_for_dexit = True
            c_high = h
            c_idx  = i
            continue

        # ── Candle DExit: open ≈ low (doji-type, open at low) ──
        candle_dexit = wait_for_dexit and (abs(o - l) < MIN_TICK)
        if candle_dexit:
            wait_for_dexit  = False
            wait_for_dentry = True
            highest_wick_d  = h
            dexit_low = l
            dexit_idx = i
            continue

        if wait_for_dentry:
            highest_wick_d = max(highest_wick_d, h)

        # ── Candle DEntry: close breaks above previous bar's high ──
        candle_dentry = wait_for_dentry and (c > highs[i - 1])
        if candle_dentry:
            result = {
                "found":        True,
                "dEntryLine":   l,
                "dentry_index": i,
                "a_index":      a_idx,
                "b_index":      b_idx,
                "c_index":      c_idx,
                "dexit_index":  dexit_idx,
                "a_high":       a_high,
                "b_low":        b_low,
                "c_high":       c_high,
                "dexit_low":    dexit_low,
            }
            wait_for_dentry = False
            highest_wick    = np.nan
            highest_wick_d  = np.nan

    return result


# ─────────────────────────────────────────
#  FETCH + SCAN ONE SYMBOL
# ─────────────────────────────────────────
def scan_symbol(symbol: str) -> dict:
    try:
        raw = yf.download(
            symbol,
            interval=INTERVAL,
            period=PERIOD,
            progress=False,
            auto_adjust=True
        )

        # ── Guard: empty dataframe ──
        if raw is None or raw.empty:
            return {"symbol": symbol, "status": "NO_DATA",
                    "reason": "Empty dataframe — ticker may be wrong or delisted"}

        # Flatten MultiIndex columns (yfinance >= 0.2.x returns MultiIndex)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Keep only OHLCV
        raw = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        raw.columns = ["open", "high", "low", "close", "volume"]
        raw.dropna(inplace=True)

        if len(raw) < 10:
            return {"symbol": symbol, "status": "NO_DATA",
                    "reason": f"Only {len(raw)} rows — not enough data"}

        ha_df = compute_heikin_ashi(raw)
        res   = detect_structure(ha_df)

        current_price    = float(ha_df["close"].iloc[-1])
        last_candle_time = raw.index[-1]

        if not res["found"]:
            return {
                "symbol":        symbol,
                "status":        "NO_STRUCTURE",
                "current_price": current_price,
                "last_candle":   last_candle_time,
            }

        d_line   = res["dEntryLine"]
        distance = abs(current_price - d_line)
        alert    = distance <= ALERT_DISTANCE

        return {
            "symbol":        symbol,
            "status":        "ALERT" if alert else "WATCHING",
            "current_price": current_price,
            "dEntryLine":    d_line,
            "distance":      distance,
            "alert":         alert,
            "last_candle":   last_candle_time,
            "structure":     res,
        }

    except Exception as e:
        return {"symbol": symbol, "status": "ERROR", "error": str(e)}


# ─────────────────────────────────────────
#  TELEGRAM NOTIFICATION
# ─────────────────────────────────────────
def send_telegram_message(message: str):
    """Sends a notification to the configured Telegram chat."""
    if not TELEGRAM_BOT_TOKEN or "YOUR_BOT_TOKEN" in TELEGRAM_BOT_TOKEN:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            print(f"  [!] Telegram API Error: {response.status_code} - {response.text}")
        else:
            print("  [✓] Telegram notification sent successfully.")
    except Exception as e:
        print(f"  [!] Failed to send Telegram alert: {e}")


# ─────────────────────────────────────────
#  PRINT RESULTS
# ─────────────────────────────────────────
def print_results(results: list):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 65)
    print(f"  Nifty 10  |  Timeframe: 1H  |  Scan Time: {now}")
    print("=" * 65)

    alerts   = [r for r in results if r.get("status") == "ALERT"]
    watching = [r for r in results if r.get("status") == "WATCHING"]
    no_str   = [r for r in results if r.get("status") == "NO_STRUCTURE"]
    errors   = [r for r in results if r.get("status") in ("ERROR", "NO_DATA")]

    # ── ALERTS ──
    if alerts:
        print(f"\n  *** ALERTS ({len(alerts)} stock{'s' if len(alerts) > 1 else ''}) ***\n")
        
        # Prepare Telegram Message
        tg_msg = f"🚨 *ABC-DEntry Alerts ({len(alerts)})*\n"
        tg_msg += f"Scan Time: {now}\n\n"

        for r in alerts:
            print(f"  +-- {r['symbol']}")
            print(f"  |   Current Price : Rs.{r['current_price']:.2f}")
            print(f"  |   dEntryLine    : Rs.{r['dEntryLine']:.2f}")
            print(f"  |   Distance      : {r['distance']:.2f} pts  <-- WITHIN {ALERT_DISTANCE} POINTS!")
            s = r["structure"]
            print(f"  |   Pattern       : A={s['a_high']:.2f}  B={s['b_low']:.2f}  "
                  f"C={s['c_high']:.2f}  DExit={s['dexit_low']:.2f}")
            print(f"  +-- Last Candle   : {r['last_candle']}\n")

            tg_msg += (f"🔹 *{r['symbol']}*\n"
                       f"Price: Rs.{r['current_price']:.2f}\n"
                       f"Line: Rs.{r['dEntryLine']:.2f}\n"
                       f"Dist: {r['distance']:.2f} pts\n\n")
            
        # Send batch notification
        send_telegram_message(tg_msg)

    else:
        print("\n  [OK] No alerts firing right now.\n")

    # ── WATCHING ──
    if watching:
        print(f"  [WATCHING] Structure found, price not yet near line\n")
        print(f"  {'Symbol':<16} {'Price':>10} {'dEntryLine':>12} {'Distance':>12}")
        print(f"  {'-'*16} {'-'*10} {'-'*12} {'-'*12}")
        for r in watching:
            print(f"  {r['symbol']:<16} Rs.{r['current_price']:>8.2f}"
                  f"  Rs.{r['dEntryLine']:>9.2f}  {r['distance']:>8.2f} pts")
        print()

    # ── NO STRUCTURE ──
    if no_str:
        syms = ", ".join(r["symbol"] for r in no_str)
        print(f"  [PENDING] No complete structure yet: {syms}\n")

    # ── ERRORS / NO DATA ──
    if errors:
        print(f"  [ERROR] Failed to fetch:")
        for r in errors:
            reason = r.get("error") or r.get("reason") or r["status"]
            print(f"    {r['symbol']} --> {reason}")
        print()

    print("=" * 65)
    print(f"  Total: {len(alerts)} alert(s)  |  {len(watching)} watching  |"
          f"  {len(no_str)} pending  |  {len(errors)} error(s)")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
def main():
    print("\n" + "=" * 65)
    print("  A-B-C-DExit-DEntry Scanner  |  Heikin Ashi  |  1H TF")
    print("=" * 65)
    print(f"  Stocks  : {len(NIFTY_10)} Nifty stocks")
    print(f"  Alert   : Price within {ALERT_DISTANCE} points of dEntryLine")
    print(f"  Data    : Last {PERIOD} @ {INTERVAL} interval")
    print("=" * 65 + "\n")

    # Test notification to verify connection on startup
    send_telegram_message(f"🚀 *ABC-DEntry Scanner Started*\nMonitoring {len(NIFTY_10)} stocks...")

    results = []
    for symbol in NIFTY_10:
        print(f"  Scanning {symbol:<18}", end="", flush=True)
        res = scan_symbol(symbol)
        results.append(res)
        status = res["status"]
        if status == "ALERT":
            print(f"--> *** ALERT *** (dist={res['distance']:.1f} pts)")
        elif status == "WATCHING":
            print(f"--> Watching  (dist={res['distance']:.1f} pts from line)")
        elif status == "NO_STRUCTURE":
            print(f"--> No pattern found")
        else:
            print(f"--> {status}: {res.get('error') or res.get('reason', '')}")

    print_results(results)


if __name__ == "__main__":
    main()
