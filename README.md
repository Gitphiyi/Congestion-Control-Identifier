# TCP Congestion Control Algorithm Classifier

ML system to identify TCP congestion control algorithms (Reno, BBR, BIC, HTCP, etc.) from network packet traces using polynomial feature extraction and Gaussian classification. Adapted from Nebby CCA by Mishra et. al.

## Quick Start

### Step 1: Parse PCAP to CSV
Edit configs in tcp_flow_capture/parse_trace.py to parse flow PCAP to CSV.

### Step 2: Configure Training/Testing Files
Edit `globals.py`:
```python
TRAINING_FILES = {
    'reno': ['traces/reno_1.csv', 'traces/reno_2.csv', ...],
    'bbr': ['traces/bbr_1.csv', 'traces/bbr_2.csv', ...],
    # ... add your CCAs
}
TESTING_FILES = {
    'reno': ['traces/reno_5.csv', ...],
    'bbr': ['traces/bbr_5.csv', ...],
    # ... add your CCAs
}
```

### Step 3: Train the Model
```bash
cd tcp_flow_analysis
python train.py
```
This saves the trained model to `models/cca_classifier.pkl`

### Step 4: Classify Traces
```bash
cd tcp_flow_analysis
python test.py
```
This generates a confusion matrix based on the traces provided in TESTING_FILES.