import pyshark
import pandas as pd
import asyncio

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

pcap_file = "traces/trace2.pcap"

display_filter = "tcp"

cap = pyshark.FileCapture(pcap_file, display_filter="tcp", keep_packets=True, eventloop=loop)
cap.load_packets()

print("Total packets in capture:", len(cap))

rows = []
for pkt in cap:
    rows.append({
        "time": float(pkt.sniff_timestamp),
        "src": pkt.ip.src,
        "dst": pkt.ip.dst,
        "seq": int(pkt.tcp.seq),
        "ack": int(pkt.tcp.ack),
        "len": int(pkt.tcp.len),
        "bytes_in_flight": int(getattr(pkt.tcp, "analysis_bytes_in_flight", 0))
    })

df = pd.DataFrame(rows)
print(df)

cap.close()