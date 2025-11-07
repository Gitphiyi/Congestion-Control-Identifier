#!/usr/bin/env python3
import asyncio
from datetime import datetime, timezone, timedelta
import pyshark
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional


# === CONFIG ===
PCAP_FILE = "traces/trace4.pcap"
OUT_PNG = "plots/bytes_in_flight.png"
TIMEZONE = timezone(timedelta(hours=-5))
DISPLAY_FILTER = "tcp"
KEEP_PACKETS = True
EVENT_LOOP = asyncio.new_event_loop()
# =============


def fld(obj, name):
    return getattr(obj, name, None)

def format_time(sniff_timestamp: str, tz: timezone=TIMEZONE) -> Optional[datetime]:
    try:
        ts = float(sniff_timestamp)
        return datetime.fromtimestamp(ts, tz)
    except Exception:
        return None

def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default
    
def parse_capture(capture) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pkt in capture:
        tcp = getattr(pkt, "tcp", None)
        if not tcp:
            continue
        
        # times
        sniff_ts = getattr(pkt, "sniff_timestamp", None)
        t_dt = format_time(sniff_ts) if sniff_ts is not None else None
        ip = getattr(pkt, "ip", None)
        src = getattr(ip, "src", None) if ip else None
        dst = getattr(ip, "dst", None) if ip else None

        seq = safe_int(getattr(tcp, "seq", None))
        ack = safe_int(getattr(tcp, "ack", None))
        length = safe_int(getattr(tcp, "len", None))

        bif_auto = 0
        try:
            bif_auto = int(getattr(tcp, "analysis_bytes_in_flight", 0))
        except Exception:
            bif_auto = 0

        rows.append({
            "time": t_dt,
            "src": src,
            "dst": dst,
            "seq": seq,
            "ack": ack,
            "len": length,
            "bif_auto": bif_auto
        })

    return rows

def calc_bif(df):
    if df.empty:
        return pd.Series(dtype=int)
    
    sender_ip = df.groupby("src")["len"].sum().idxmax()
    print("Heuristic sender IP:", sender_ip)

    df_sorted = df.sort_values("time", na_position="last").reset_index(drop=True)

    last_sent = 0
    last_acked = 0
    bif = []
    for _, row in df_sorted.iterrows():
        payload_end = row["seq"] + row["len"]
        if row["src"] == sender_ip and row["len"] > 0:
            last_sent = max(last_sent, payload_end)
        if row["src"] != sender_ip and row["ack"] > 0:
            last_acked = max(last_acked, row["ack"])
        bif.append(max(0, last_sent - last_acked))

    return pd.Series(bif, index=df_sorted.index).reindex(df.index).fillna(0).astype(int)

def plot_bif(df: pd.DataFrame, bif_col: str = "bytes_in_flight_calc", time_col: str = "time", out_png: str = OUT_PNG):
    if df.empty:
        print("Empty DataFrame — nothing to plot.")
        return

    if time_col not in df.columns or bif_col not in df.columns:
        raise KeyError(f"Required columns missing: {time_col} or {bif_col}")

    plot_df = df[[time_col, bif_col]].dropna().copy()
    # Ensure time is datetime
    if not pd.api.types.is_datetime64_any_dtype(plot_df[time_col]):
        plot_df[time_col] = pd.to_datetime(plot_df[time_col])

    plt.figure(figsize=(10, 4.5))
    plt.plot(plot_df[time_col], plot_df[bif_col], drawstyle="steps-post")
    plt.xlabel("Time")
    plt.ylabel("Bytes in flight")
    plt.title("Bytes-in-Flight over Time")
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.show()
    print(f"Saved plot to {out_png}")



asyncio.set_event_loop(EVENT_LOOP)

cap = pyshark.FileCapture(PCAP_FILE, display_filter=DISPLAY_FILTER, keep_packets=KEEP_PACKETS, eventloop=EVENT_LOOP)
cap.load_packets()

print("Total packets in capture:", len(cap))

rows = parse_capture(cap)

cap.close()

df = pd.DataFrame(rows)

bytes_in_flight = calc_bif(df)
df["bif_calc"] = bytes_in_flight

print("Result rows:", df.shape[0])
print(df[["time", "src", "dst", "seq", "ack", "len", "bif_auto", "bif_calc"]].head(20).to_string())

plot_bif(df, bif_col="bif_calc", time_col="time", out_png=OUT_PNG)

