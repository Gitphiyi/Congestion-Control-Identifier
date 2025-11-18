#!/usr/bin/env python3
import asyncio
from datetime import datetime, timezone, timedelta
import pyshark
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from statistics import mean

PCAP_FILE = "traces/bic_8.pcap"
OUT_PNG = "plots/bic_8.png"
TIMEZONE = timezone(timedelta(0))
DISPLAY_FILTER = "tcp"
KEEP_PACKETS = True
EVENT_LOOP = asyncio.new_event_loop()
# SENDER_IP = "18.208.88.157"
# RECEIVER_IP = "67.159.78.206"
SENDER_IP = "67.159.78.206"
RECEIVER_IP = "67.159.65.185"

END_FRAME = 15000 # IPERF TEST END
STREAM_NUMBER = 1 # 1
TOTAL_DELAY = 100 # MS (50 ms on each sender, receiver)


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

def parse_capture(capture) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    ack_rows: List[Dict[str, Any]] = []
    for pkt in capture:
        tcp = getattr(pkt, "tcp", None)
        if not tcp:
            continue

        frame_number = safe_int(getattr(pkt, "number", None))
        if frame_number > END_FRAME:
            break

        ip = getattr(pkt, "ip", None)
        src = getattr(ip, "src", None) if ip else None
        dst = getattr(ip, "dst", None) if ip else None
        stream = safe_int(getattr(tcp, "stream", None))

        if stream != STREAM_NUMBER:
            continue

        if src == SENDER_IP:
            sniff_ts = getattr(pkt, "sniff_timestamp", None)
            t_dt = format_time(sniff_ts) if sniff_ts is not None else None

            seq = safe_int(getattr(tcp, "seq", None))
            ack = safe_int(getattr(tcp, "ack", None))
            length = safe_int(getattr(tcp, "len", None))

            bif_auto = 0
            try:
                bif_auto = int(getattr(tcp, "analysis_bytes_in_flight", 0))
            except Exception:
                bif_auto = 0
        
            rows.append({
                "number": frame_number,
                "time": t_dt,
                "src": src,
                "dst": dst,
                "seq": seq,
                "ack": ack,
                "len": length,
                "stream": stream,
                "bif_auto": bif_auto
            })
        elif src == RECEIVER_IP: # this is an ack
            ack_for = getattr(tcp, "analysis_acks_frame", 0)
            rtt = getattr(tcp, "analysis_ack_rtt", -1)

            ack_rows.append({
                "number": frame_number,
                "src": src,
                "dst": dst,
                "ack_for": ack_for,
                "ack_rtt": rtt
            })

    return rows, ack_rows

def merge_ack_data(
    normal_df: pd.DataFrame,
    ack_df: pd.DataFrame,
    frame_col: str = "number",        
    ack_for_col: str = "ack_for",
    ack_rtt_col: str = "ack_rtt",
    out_col: str = "rtt",
) -> pd.DataFrame:
    # defensive copy
    df_packets = normal_df.copy()
    df_acks = ack_df.copy()

    # make sure columns exist
    if frame_col not in df_packets.columns:
        raise KeyError(f"normal_df missing required column: {frame_col}")
    if ack_for_col not in df_acks.columns or ack_rtt_col not in df_acks.columns:
        # nothing to annotate: add out_col with NaN and return
        df_packets[out_col] = pd.NA
        return df_packets

    # coerce numeric types for matching (NaN will be dropped)
    df_packets[frame_col] = pd.to_numeric(df_packets[frame_col], errors="coerce")
    df_acks[ack_for_col] = pd.to_numeric(df_acks[ack_for_col], errors="coerce")
    # keep only ack rows that actually reference a frame
    df_acks = df_acks.loc[df_acks[ack_for_col].notna()]

    # Build mapping: frame_number -> RTT
    mapping = {}
    for _, ack_row in df_acks.iterrows():
        frame = int(ack_row[ack_for_col])
        rtt_val = ack_row[ack_rtt_col]
        if pd.notna(rtt_val):
            mapping[frame] = float(rtt_val)

    # Compute a fallback RTT (mean of all valid RTTs, or NaN if none)
    rtt_values = list(mapping.values())
    avg_rtt = mean(rtt_values) if rtt_values else np.nan

    # Apply mapping; missing entries replaced with avg_rtt
    df_packets[out_col] = df_packets[frame_col].map(mapping).fillna(avg_rtt)

    return df_packets

def merge_rtt_packets_timedelta(
    df: pd.DataFrame,
    time_col: str = "time",
    bif_col: str = "bif_auto",
    total_delay_ms: Optional[float] = TOTAL_DELAY
) -> pd.DataFrame:
    df_local = df.copy()

    # Ensure time column exists and is datetime
    if time_col not in df_local.columns:
        raise KeyError(f"time column {time_col!r} not found in DataFrame")
    df_local[time_col] = pd.to_datetime(df_local[time_col])

    delay_td = pd.to_timedelta(total_delay_ms, unit="ms")

    df_local = df_local.sort_values(time_col).reset_index(drop=True)

    # compute inter-packet time deltas
    time_deltas = df_local[time_col].diff().fillna(pd.Timedelta(seconds=0))
    # mark where a new window should start: when delta > delay_td
    new_window = (time_deltas > delay_td).astype(int)
    # create window ids by cumulative sum of new_window
    window_id = new_window.cumsum()
    df_local["_window_id"] = window_id
    # aggregate per window
    agg = df_local.groupby("_window_id").agg(
        start=(time_col, "first"),
        end=(time_col, "last"),
        count=(time_col, "count"),
        sum_bif=(bif_col, lambda s: s.fillna(0).sum()),
        mean_bif=(bif_col, lambda s: s.fillna(0).mean()),
        max_bif=(bif_col, lambda s: s.fillna(0).max())
    ).reset_index(drop=False)

    agg["duration_s"] = (agg["end"] - agg["start"]).dt.total_seconds()
    agg["delay_used_ms"] = total_delay_ms
    agg = agg.rename(columns={"_window_id": "window_id"})

    # reorder columns
    agg = agg[["window_id", "start", "end", "count", "duration_s", "sum_bif", "mean_bif", "max_bif", "delay_used_ms"]]

    return agg

def merge_packets_fixed_window_start(
    df: pd.DataFrame,
    time_col: str = "time",
    bif_col: str = "bif_auto",
    window_ms: float = 100.0
) -> pd.DataFrame:
    if df is None or df.shape[0] == 0:
        return pd.DataFrame(columns=[
            "window_id", "start", "end", "count", "duration_s",
            "sum_bif", "mean_bif", "max_bif", "delay_used_ms"
        ])

    df_local = df.copy()
    if time_col not in df_local.columns:
        raise KeyError(f"time column {time_col!r} not found in DataFrame")
    df_local[time_col] = pd.to_datetime(df_local[time_col])

    if bif_col not in df_local.columns:
        df_local[bif_col] = 0
    df_local[bif_col] = df_local[bif_col].fillna(0)

    df_local = df_local.sort_values(time_col).reset_index(drop=True)

    try:
        window_ms = float(window_ms)
    except Exception:
        window_ms = 100.0
    window_td = pd.to_timedelta(window_ms, unit="ms")

    # iterate rows and build windows
    windows = []
    cur_start = df_local.at[0, time_col]
    cur_end = cur_start
    cur_sum = float(df_local.at[0, bif_col])
    cur_count = 1
    cur_max = float(df_local.at[0, bif_col])

    for idx in range(1, len(df_local)):
        t = df_local.at[idx, time_col]
        bif_val = float(df_local.at[idx, bif_col])

        if t <= cur_start + window_td:
            cur_end = t
            cur_count += 1
            cur_sum += bif_val
            if bif_val > cur_max:
                cur_max = bif_val
        else:
            windows.append({
                "start": cur_start,
                "end": cur_end,
                "count": cur_count,
                "sum_bif": cur_sum,
                "mean_bif": (cur_sum / cur_count) if cur_count > 0 else 0.0,
                "max_bif": cur_max
            })
            cur_start = t
            cur_end = t
            cur_sum = bif_val
            cur_count = 1
            cur_max = bif_val

    # append final window
    windows.append({
        "start": cur_start,
        "end": cur_end,
        "count": cur_count,
        "sum_bif": cur_sum,
        "mean_bif": (cur_sum / cur_count) if cur_count > 0 else 0.0,
        "max_bif": cur_max
    })

    # build result DataFrame
    agg = pd.DataFrame(windows)
    agg["duration_s"] = (agg["end"] - agg["start"]).dt.total_seconds()
    agg["delay_used_ms"] = float(window_ms)
    agg.insert(0, "window_id", range(len(agg)))

    # reorder columns for readability
    agg = agg[["window_id", "start", "end", "count", "duration_s", "sum_bif", "mean_bif", "max_bif", "delay_used_ms"]]

    return agg

def plot_bif_time(df: pd.DataFrame, bif_col: str = "bif_auto", time_col: str = "time", out_png: str = OUT_PNG):
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

def plot_bif_window(df: pd.DataFrame,
             bif_col: str = "mean_bif",
             x_col: str = "window_id",
             out_png: str = OUT_PNG):
    if df.empty:
        print("Empty DataFrame — nothing to plot.")
        return

    if x_col not in df.columns or bif_col not in df.columns:
        raise KeyError(f"Required columns missing: {x_col} or {bif_col}")

    plot_df = df[[x_col, bif_col]].dropna().copy()

    plt.figure(figsize=(10, 4.5))
    plt.plot(plot_df[x_col], plot_df[bif_col], drawstyle="steps-post")
    plt.xlabel("RTT / Window index")
    plt.ylabel("Bytes in flight")
    plt.title(f"Bytes-in-Flight per RTT window ({bif_col})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.show()
    print(f"Saved plot to {out_png}")


asyncio.set_event_loop(EVENT_LOOP)

cap = pyshark.FileCapture(PCAP_FILE, display_filter=DISPLAY_FILTER, keep_packets=KEEP_PACKETS, eventloop=EVENT_LOOP)
cap.load_packets()

print("Total packets in capture:", len(cap))

rows, ack_rows = parse_capture(cap)
cap.close()

df = pd.DataFrame(rows)
ack_df = pd.DataFrame(ack_rows)

df = merge_ack_data(df, ack_df)

# df["time"] = pd.to_datetime(df["time"])
# # set as index, then resample
# df = df.set_index("time").sort_index()
df["time"] = pd.to_datetime(df["time"])

median_rtt = df["rtt"].dropna().median()
print(f"Mean RTT = {median_rtt*1000:.2f} ms")

# print(merged_df[["number", "src", "dst", "len", "rtt"]].head(10).to_string())
# print(df[["number", "time", "src", "dst", "seq", "ack", "len", "rtt", "bif_auto"]].head(40).to_string())

# agg_df = merge_rtt_packets_timedelta(df, "time", "bif_auto", TOTAL_DELAY)
agg_df = merge_packets_fixed_window_start(df, "time", "bif_auto", TOTAL_DELAY)
# print(agg_df[["window_id", "start", "end", "count", "duration_s", "sum_bif", "mean_bif", "max_bif", "delay_used_ms"]].head(40).to_string())

# plot_bif_window(agg_df, bif_col="mean_bif", x_col="window_id", out_png=OUT_PNG)
# plot_bif_time(df.iloc[10:800], "bif_auto", "time", OUT_PNG)

#  skip slow start
plot_bif_window(agg_df.iloc[10:100], bif_col="mean_bif", x_col="window_id", out_png=OUT_PNG)