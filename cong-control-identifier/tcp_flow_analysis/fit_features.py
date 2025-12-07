import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
from globals import FEATURE_NUM, MAX_DEG

"""
polynomial fitting for features:

1. load BiF csv from parse_trace.py output
2. apply smoothing
3. extract features
4. normalize (by max of feature) and fit polynomials
"""

def load_csv_simple(csv_path):
    """
    returns:
        time: list of relative timestamps (seconds)
        data: list of BiF values (bif_auto)
        retrans: list of retransmission timestamps
        measured_rtt: median RTT from csv (seconds)
    """
    df = pd.read_csv(csv_path)

    # convert time to relative seconds
    df['time'] = pd.to_datetime(df['time'])
    start_time = df['time'].min()
    df['time_rel'] = (df['time'] - start_time).dt.total_seconds()

    # sort by time (just in case)
    df = df.sort_values('time_rel')

    # extract data
    time = df['time_rel'].tolist()
    data = df['bif_auto'].fillna(0).astype(int).tolist()

    # get retransmissions (seq < max_seq)
    retrans = []
    max_seq = 0
    for idx, row in df.iterrows():
        seq = int(row['seq']) if pd.notna(row['seq']) else 0
        if seq > 0 and seq < max_seq:
            retrans.append(float(row['time_rel']))
        max_seq = max(max_seq, seq)

    # get RTT (median)
    rtt_values = df['rtt'].dropna()
    measured_rtt = float(rtt_values.median())

    print(f'NUM RETRANSMISSIONS: {len(retrans)}')
    return time, data, retrans, measured_rtt


def smoothen(time, data, rtt):
    # rolling window smoothing (from Nebby)
    left = 0
    right = 0
    run_sum = 0
    avg_data = []
    new_time = []

    while right < len(time):
        while(right < len(time) and (time[right]-time[left] < 2*rtt)):
            run_sum += data[right]
            right += 1
        new_time.append(float(time[right-1]+time[left])/2)
        avg_data.append(float(run_sum)/(right-left))
        run_sum -= data[left]
        left += 1

    return new_time, avg_data


def get_time_features(retrans, time, rtt):
    # get feature windows from retransmissions
    time_thresh = 20*rtt
    features = []

    if len(retrans) < 2:
        # if no retransmissions, use entire trace as one feature
        return [[time[0], time[-1]]]

    for i in range(1, len(retrans)):
        if retrans[i]-retrans[i-1] >= time_thresh:
            features.append([retrans[i-1], retrans[i]])

    # add last feature from last retrans to end
    features.append([retrans[-1], time[-1]])

    print('NUM FEATURES (ACTUAL): ', len(features))
    return features


def get_features(time, time_features):
    # convert time-based features to index-based
    left = 0
    right = 0
    feature_index = 0
    in_feature = 0
    index_features = []

    while right < len(time) and feature_index < len(time_features):
        if in_feature == 0 and time[right] >= time_features[feature_index][0]:
            in_feature = 1
            left = right
        elif in_feature == 1 and time[right] > time_features[feature_index][1]:
            in_feature = 0
            index_features.append([left, right-1])
            feature_index += 1
        right += 1

    if in_feature == 1:
        index_features.append([left, right-1])

    return index_features


def normalize(time, data, rtt):
    time = np.array(time)
    data = np.array(data)

    # normalize
    new_time = time / rtt
    new_data = (data / np.max(data)) * 100  # normalize by max of current feature

    # center around origin
    new_time -= np.min(new_time)
    new_data -= np.min(new_data)

    return new_time, new_data


def get_degree(time, data, p="n", max_deg=MAX_DEG):
    """
    fit polynomials of degree 1 to max_deg and return coefficients.

    args:
        time: normalized time array
        data: normalized BIF array
        max_deg: max polynomial degree (default from globals)

    returns:
        (degree, coefficients, mse_list)
    """
    time = np.array(time)
    data = np.array(data)

    p_net = []
    mse_l = []
    fit_net = []

    for d in range(1, max_deg + 1):
        p_temp = np.polyfit(time, data, d)
        p_net.append(p_temp)
        fit_net.append(np.polyval(p_temp, time))
        mse_l.append(mse(data, fit_net[-1]))

    if p == 'y':
        plt.figure(figsize=(10, 6))
        plt.plot(time, data, 'ko-', label='Actual', markersize=4)
        for d in range(max_deg):
            plt.plot(time, fit_net[d], label=f'Degree {d+1}', alpha=0.7)
        plt.xlabel('Normalized Time (time/RTT)')
        plt.ylabel('Normalized BIF (% of max)')
        plt.title('Polynomial Fits')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return max_deg, p_net[max_deg - 1], mse_l


# MAIN PIPELINE
def process_single_feature(csv_path, feature_num=FEATURE_NUM, max_deg=MAX_DEG, plot=False):
    """
    args:
        csv_path: Path to CSV from parse_trace.py (must contain 'rtt' column)
        feature_num: Which feature to extract (default from globals)
        max_deg: Maximum polynomial degree (default from globals)
        plot: Whether to plot the fit

    returns:
        map: 'degree', 'coeff', 'error', 'data', 'time', 'rtt'
    """

    # Step 1: load CSV
    time, data, retrans, rtt = load_csv_simple(csv_path)

    # Step 2: smooth
    smooth_time, smooth_data = smoothen(time, data, rtt)

    # Step 3: get features
    time_features = get_time_features(retrans, smooth_time, rtt)
    index_features = get_features(smooth_time, time_features)
    if len(index_features) == 0:
        raise ValueError(f"No features found in {csv_path}")
    if feature_num >= len(index_features):
        print(f"Warning: Feature {feature_num} doesn't exist, using feature 0")
        feature_num = 0

    # Step 4: get specific feature
    start_idx, end_idx = index_features[feature_num]
    feature_time = smooth_time[start_idx:end_idx+1]
    feature_data = smooth_data[start_idx:end_idx+1]

    # Step 4.5: trim to min-max range (remove tails)
    min_idx = np.argmin(feature_data)
    max_idx = np.argmax(feature_data)
    trim_start = min(min_idx, max_idx)
    trim_end = max(min_idx, max_idx)
    feature_time = feature_time[trim_start:trim_end+1]
    feature_data = feature_data[trim_start:trim_end+1]

    # Step 5: normalize
    norm_time, norm_data = normalize(feature_time, feature_data, rtt)

    # Step 6: fit polynomial
    degree, coeff, error = get_degree(norm_time, norm_data, p="y" if plot else "n", max_deg=max_deg)

    return {
        'degree': degree,
        'coeff': coeff,
        'error': error,
        'data': norm_data,
        'time': norm_time,
        'rtt': rtt,
    }


# EXAMPLE USAGE
if __name__ == "__main__":
    csv_path = "../tcp_flow_capture/traces/parsed/cubic_1.csv"

    print("Testing polynomial fitting...")
    print("="*60)

    result = process_single_feature(
        csv_path=csv_path,
        plot=True
    )

    print("\nResults:")
    print(f"  Degree: {result['degree']}")
    print(f"  Coefficients: {result['coeff']}")
    print(f"  MSE per degree: {result['error']}")
    print(f"  RTT: {result['rtt']*1000:.2f} ms")
