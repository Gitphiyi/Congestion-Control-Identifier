# Client-side CCA Inference

## Running the identifier

Change host url (and optional endpoint) in `client.py`

`poetry run python client.py`

`sudo tcpdump -i en0 tcp and host www.harrisonyork.com -w traces/trace.pcap`



## Running data transform

Change trace path `traces/example.pcap`

`poetry run python transform.py`