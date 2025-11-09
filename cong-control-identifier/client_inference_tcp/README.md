# Client-side CCA Inference

## Running and capturing TCP flow

Start Linux VMs, designate one as sender (client) and one as receiver (server).

On receiver:
```
export IF=eth0
sudo tc qdisc add dev $IF root netem delay 50ms rate 10mbit
iperf3 -s
```
Set the interface as eth0 and add a 50 ms delay for TCP with max bandwidth of 10 megabits per second. Then start iperf.


On sender:
```
sudo sysctl -w net.ipv4.tcp_congestion_control=cubic
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr
sudo sysctl -w net.ipv4.tcp_congestion_control=reno
```
Choose the CCA to use.
```
IF=eth0
sudo tc qdisc add dev $IF root netem delay 50ms rate 10mbit
sudo tcpdump -i $IF -s 0 tcp and port 5201 -w traces/test_output.pcap -U
```
Also add a 50 ms delay here, which does not factor into the RTT but separates round trips by 50 ms. Use tcpdump to track the TCP flow and save it in specified output file.

In a separate terminal on sender:
```
iperf3 -c <server-ip> -t 15
```
For the existing tests: 
- Server IP = 67.159.65.185
- Client IP = 67.159.78.206

When the test ends, end the tcpdump process and parse the trace.

To remove the queueing discipline that controls the delay and limits rate:
```
sudo tc qdisc del dev $IF root
```
- tc = Traffic Control
- qdisc = Queueing discipline (how packets are queued / scheduled)



## Parsing traces
In parse_trace.py

Update variables:
- PCAP_FILE
- OUT_PNG
- END_FRAME (can find via Wireshark)

`poetry run python parse_trace.py`

Output will be a plot of BiF vs Time. 

Currently the numbers make sense: about 60000 bytes in flight = 480,000 bits in flight per 50 ms. 

50 ms * 20 = 1 sec, so 480,000 bits per 50 ms = 9,600,000 bits per 1 second. Just under 10 mbits per second. 