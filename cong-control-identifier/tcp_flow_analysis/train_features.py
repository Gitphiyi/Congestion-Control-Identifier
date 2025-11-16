import numpy as np
import pickle
from fit_features import process_single_feature
from globals import BANDWIDTH_KBPS, BDP_FACTOR, FEATURE_NUM, MAX_DEG, REGULARIZATION

"""
Build Gaussian classifier from CSV training files
"""


def train_classifier(csv_files_by_cca, bandwidth_kbps=BANDWIDTH_KBPS, bdp_factor=BDP_FACTOR,
                     feature_num=FEATURE_NUM, max_deg=MAX_DEG):
    """
    Args:
        csv_files_by_cca: Dict of {cca_name: [list of CSV paths]}
        bandwidth_kbps: Bandwidth in kbps (default: 5000 = 5 Mbit/s)
        bdp_factor: BDP factor
        feature_num: Which feature to use (0 = first)
        max_deg: Maximum polynomial degree

    Returns:
        gaussian_params: Dict of {cca_name: {'mean': [...], 'covar': [...]}}
    """

    print("="*80)
    print("TRAINING GAUSSIAN CLASSIFIER")
    print("="*80)

    # get coefficients for each CCA
    cca_coefficients = {}

    for cca_name, csv_list in csv_files_by_cca.items():
        print(f"\nProcessing {cca_name}: {len(csv_list)} files")
        cca_coefficients[cca_name] = []

        for csv_path in csv_list:
            try:
                result = process_single_feature(
                    csv_path=csv_path,
                    bandwidth_kbps=bandwidth_kbps,
                    bdp_factor=bdp_factor,
                    feature_num=feature_num,
                    max_deg=max_deg,
                    plot=False
                )
                cca_coefficients[cca_name].append(result['coeff'])
                print(f"  DONE: {csv_path}")

            except Exception as e:
                print(f"  ERROR: {csv_path}: {e}")

    # compute Gaussian params (mean + diagonal covariance)
    gaussian_params = {}

    print("\n" + "="*80)
    print("COMPUTING GAUSSIAN PARAMETERS")
    print("="*80)

    for cca_name, coeffs_list in cca_coefficients.items():
        if len(coeffs_list) == 0:
            print(f"\n{cca_name}: No valid features, skipping")
            continue

        # stack coefficients
        coeffs_array = np.array(coeffs_list)  # shape: (num_files, num_coeffs)

        # get mean
        mean = np.mean(coeffs_array, axis=0)

        # get covariance
        covar_full = np.cov(coeffs_array, rowvar=False)

        # use diag covariance only (like Nebby)
        identity = np.identity(len(mean))
        covar_diag = covar_full * identity

        # add regularization to prevent singular matrices
        # this adds a small value to diagonal elements to ensure numerical stability
        covar_diag_reg = covar_diag + REGULARIZATION * identity

        gaussian_params[cca_name] = {
            'mean': mean,
            'covar': covar_diag_reg
        }

        print(f"\n{cca_name}:")
        print(f"  Training samples: {len(coeffs_list)}")
        print(f"  Mean coefficients: {mean}")
        print(f"  Covariance diagonal (before reg): {np.diag(covar_diag)}")
        print(f"  Covariance diagonal (after reg): {np.diag(covar_diag_reg)}")
        print(f"  Regularization added: {REGULARIZATION}")

    print("\n" + "="*80)
    print(f"TRAINING COMPLETE: {len(gaussian_params)} CCAs")
    print("="*80)

    return gaussian_params


def save_model(gaussian_params, filename="cca_model.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(gaussian_params, f)


def load_model(filename="cca_model.pkl"):
    with open(filename, 'rb') as f:
        gaussian_params = pickle.load(f)
    return gaussian_params


# EXAMPLE USAGE
if __name__ == "__main__":
    # Example training data
    csv_files_by_cca = {
        'reno': [
            '../tcp_flow_capture/traces/parsed/reno_1.csv',
        ],
        'cubic': [
            '../tcp_flow_capture/traces/parsed/cubic_1.csv',
            '../tcp_flow_capture/traces/parsed/cubic_2.csv',
        ],
    }

    # train classifier
    gaussian_params = train_classifier(
        csv_files_by_cca=csv_files_by_cca
    )

    # save model
    save_model(gaussian_params, "tcp_flow_analysis/models/cca_classifier.pkl")