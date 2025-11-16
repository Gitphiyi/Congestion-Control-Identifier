"""
Global config for CCA classifier

Keep consistent between training and testing
"""

# ============================================================================
# NETWORK PARAMETERS
# ============================================================================
BANDWIDTH_KBPS = 5000  # 5 Mbit/s link bandwidth
BDP_FACTOR = 2         # BDP multiplication factor (measured from actual traces)

# ============================================================================
# FEATURE EXTRACTION PARAMETERS
# ============================================================================
FEATURE_NUM = 0        # Which feature to use (0 = first, most stable)
MAX_DEG = 5            # Maximum polynomial degree for fitting

# ============================================================================
# PATHS
# ============================================================================
# Base paths (relative to tcp_flow_analysis/ directory)
TRACES_PATH = "../tcp_flow_capture/traces/parsed/"
MODEL_PATH = "models/cca_classifier.pkl"

# Training files (1-4 for each CCA)
TRAINING_FILES = {
    'reno': [
        '../tcp_flow_capture/traces/parsed/reno_1.csv',
        '../tcp_flow_capture/traces/parsed/reno_2.csv',
        '../tcp_flow_capture/traces/parsed/reno_3.csv',
        '../tcp_flow_capture/traces/parsed/reno_4.csv',
    ],
    'cubic': [
        '../tcp_flow_capture/traces/parsed/cubic_1.csv',
        '../tcp_flow_capture/traces/parsed/cubic_2.csv',
        '../tcp_flow_capture/traces/parsed/cubic_3.csv',
        '../tcp_flow_capture/traces/parsed/cubic_4.csv',
    ],
    'vegas': [
        '../tcp_flow_capture/traces/parsed/vegas_1.csv',
        '../tcp_flow_capture/traces/parsed/vegas_2.csv',
        '../tcp_flow_capture/traces/parsed/vegas_3.csv',
        '../tcp_flow_capture/traces/parsed/vegas_4.csv',
    ],
}

# Testing files (file 5 for each CCA)
TESTING_FILES = {
    'reno': [
        '../tcp_flow_capture/traces/parsed/reno_5.csv',
    ],
    'cubic': [
        '../tcp_flow_capture/traces/parsed/cubic_5.csv',
    ],
    'vegas': [
        '../tcp_flow_capture/traces/parsed/vegas_5.csv',
    ],
}

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
REGULARIZATION = 1e-6  # Regularization for covariance matrix
