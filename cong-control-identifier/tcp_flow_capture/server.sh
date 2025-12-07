# !/bin/bash

mbit="mbit"
ms="ms"

CCA=$1
IF=eth0
LOSS="$2${mbit}"
DELAY="$3${ms}"
BW=5mbit
TIME=30
PKT=1K

sudo sysctl -w net.ipv4.tcp_congestion_control=$CCA

sudo tc qdisc del dev $IF root
sudo tc qdisc add dev $IF root netem delay $DELAY loss $LOSS rate $BW
