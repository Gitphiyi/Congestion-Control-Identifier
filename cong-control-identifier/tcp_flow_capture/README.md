# TCP Flow Capturing

## Using automated scripts
`sudo su` Need superuser to edit network configuration.

client.sh: `./client.sh <CCA> <loss_%> <delay_ms> <capture_file> <test_count>`

server.sh: `./server.sh <CCA> <loss_%> <delay_ms>`


## Manually running and capturing TCP flow

Start Linux VMs, designate one as sender (client) and one as receiver (server).

On receiver:
```
export IF=eth0
sudo tc qdisc add dev $IF root netem delay 50ms rate 10mbit
iperf3 -s
```
Set the interface as eth0 and add a 50 ms delay for TCP with max bandwidth of 10 megabits per second. Then start iperf.

Additional test templates (receiver)
```
                                        delay   jitter  loss    max bandwidth
sudo tc qdisc add dev $IF root netem delay 50ms 1ms loss 1% rate 5mbit
sudo tc qdisc add dev $IF root netem delay 50ms 1ms loss 0.3% rate 5mbit
```

On sender:
```
sudo sysctl -w net.ipv4.tcp_congestion_control=cubic
available: reno cubic bbr vegas westwood bic htcp veno yeah lp illinois dctcp
```
Choose the CCA to use.


```
export IF=eth0
sudo tc qdisc add dev $IF root netem delay 50ms 1ms loss 0.3% rate 5mbit
sudo tcpdump -i $IF -s 0 tcp and port 5201 -w traces/test_output.pcap -U
```
Also add a 50 ms delay here, which does not factor into the RTT but separates round trips by 50 ms. Use tcpdump to track the TCP flow and save it in specified output file.

In a separate terminal on sender:
```
iperf3 -c <server-ip> -t 15
```
Force packets to be smaller:
```
iperf3 -c <server_ip> -t 30 -l 1K
iperf3 -c 67.159.65.185 -t 30 -l 1K
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

## Notes
If poetry not found, need to update PATH:
`export PATH="$HOME/.local/bin:$PATH"`