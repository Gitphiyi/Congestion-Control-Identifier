# !/bin/bash

if [ $# -ne 5 ]; then
    echo "Usage: $0 <cca> <loss_percent> <delay_ms> <output_name> <test_count>"
    exit 1
fi

ms="ms"
CCA=$1
IF=eth0
LOSS="${2}%"
DELAY="${3}ms"
BW=5mbit
TIME=30
PKT=1K
TEST_COUNT=$5
OUTPUT_BASE=$4


cleanup() {
    sudo tc qdisc del dev $IF root 2>/dev/null
}
trap cleanup EXIT

sudo -v
sudo sysctl -w net.ipv4.tcp_congestion_control=$CCA

sudo tc qdisc del dev $IF root 2>/dev/null
sudo tc qdisc add dev $IF root netem delay $DELAY loss $LOSS rate $BW

for (( i=0; i<TEST_COUNT; i++ )); do
    OUTPUT="${OUTPUT_BASE}_${i}"

    TCPDUMP_CMD="sudo tcpdump -i ${IF} -s 0 'tcp and port 5201' -w traces/${OUTPUT}.pcap -U"

    # start tcpdump in background with timeout
    timeout 35s sudo tcpdump -i ${IF} -s 0 'tcp and port 5201' -w traces/${OUTPUT}.pcap -U &

    # run the test (30 seconds)
    iperf3 -c 67.159.65.185 -t $TIME -l $PKT

    # wait for tcpdump timeout to finish
    sleep 10 
done