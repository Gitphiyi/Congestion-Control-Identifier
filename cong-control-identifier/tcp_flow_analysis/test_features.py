import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvn
from fit_features import process_single_feature
from train_features import load_model
from globals import FEATURE_NUM, MAX_DEG

"""
Testing/Classification!
"""


def classify_trace(csv_path, gaussian_params, feature_num=FEATURE_NUM, max_deg=MAX_DEG):
    """
    classify a single csv trace

    args:
        csv_path: path to csv file to classify
        gaussian_params: trained Gaussian parameters from train_features.py

    returns:
        dict with keys: 'predicted_cca', 'probabilities', 'top3'
    """

    # get polynomial coefficients from the trace
    result = process_single_feature(
        csv_path=csv_path,
        feature_num=feature_num,
        max_deg=max_deg,
        plot=False
    )

    coeffs = result['coeff']

    # compute probability under each CCA's Gaussian model
    probabilities = {}
    for cca_name, params in gaussian_params.items():
        mean = params['mean']
        covar = params['covar']

        print(f"\n{cca_name.upper()} Gaussian Model:")
        print(f"  Mean coeffs: {mean}")
        print(f"  Covar diagonal: {np.diag(covar)}")
        print(f"  Distance from mean: {np.linalg.norm(coeffs - mean):.4f}")

        # get probability density
        prob = mvn.pdf(coeffs, mean=mean, cov=covar, allow_singular=True)
        probabilities[cca_name] = prob
        print(f"  Raw PDF value: {prob:.6e}")

    # normalize probabilities
    prob_array = np.array(list(probabilities.values()))

    print(f"\n{'='*60}")
    print(f"NORMALIZATION:")
    print(f"  Raw probs: {prob_array}")
    print(f"  Min: {np.min(prob_array):.6e}, Max: {np.max(prob_array):.6e}")
    print(f"  Range (max-min): {np.max(prob_array) - np.min(prob_array):.6e}")

    if np.max(prob_array) > 0 and (np.max(prob_array) - np.min(prob_array)) > 1e-10:
        prob_norm = (prob_array - np.min(prob_array)) / (np.max(prob_array) - np.min(prob_array))
    else:
        print(f"  WARNING: Range too small or zero, using raw probabilities")
        prob_norm = prob_array / np.sum(prob_array) if np.sum(prob_array) > 0 else prob_array

    probabilities_norm = {cca: prob_norm[i] for i, cca in enumerate(probabilities.keys())}
    print(f"  Normalized: {prob_norm}")
    print(f"{'='*60}")

    # sort by probability
    sorted_cca = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    # get top 3
    top3 = [(cca, prob, probabilities_norm[cca]) for cca, prob in sorted_cca[:3]]

    return {
        'predicted_cca': sorted_cca[0][0],
        'probabilities': probabilities,
        'probabilities_norm': probabilities_norm,
        'top3': top3
    }


def test_accuracy(csv_files_by_cca, gaussian_params, feature_num=FEATURE_NUM, max_deg=MAX_DEG):
    """
    test accuracy on multiple files and generate confusion matrix

    args:
        csv_files_by_cca: map of {true_cca: [csv_paths]}
        gaussian_params: trained model

    returns:
        confusion_df: df with confusion matrix
    """

    print("="*80)
    print("TESTING CLASSIFIER")
    print("="*80)

    # collect predictions
    results = {}
    for true_cca, csv_list in csv_files_by_cca.items():
        print(f"\nTesting {true_cca}: {len(csv_list)} files")
        results[true_cca] = []

        for csv_path in csv_list:
            try:
                classification = classify_trace(
                    csv_path=csv_path,
                    gaussian_params=gaussian_params,
                    feature_num=feature_num,
                    max_deg=max_deg
                )
                results[true_cca].append(classification)
                pred_cca = classification['predicted_cca']
                correct = "✓" if pred_cca == true_cca else "✗"
                print(f"  {correct} {csv_path}: predicted={pred_cca}")

            except Exception as e:
                print(f"  ✗ {csv_path}: {e}")

    # build confusion matrix
    print("\n" + "="*80)
    print("CONFUSION MATRIX")
    print("="*80)

    cca_names = list(csv_files_by_cca.keys())
    matrix = []

    for true_cca in cca_names:
        row = []
        total = len(results[true_cca])

        for pred_cca in cca_names:
            count = sum(1 for r in results[true_cca] if r['predicted_cca'] == pred_cca)
            row.append(count)

        # add accuracy
        if total > 0:
            correct = sum(1 for r in results[true_cca] if r['predicted_cca'] == true_cca)
            accuracy = (correct / total) * 100
            row.append(f"{accuracy:.1f}%")
        else:
            row.append("N/A")

        matrix.append(row)

    # make df
    columns = cca_names + ['Accuracy']
    df = pd.DataFrame(matrix, index=cca_names, columns=columns)

    print("\nRows = True CCA, Columns = Predicted CCA")
    print(df)

    return df


# EXAMPLE USAGE
if __name__ == "__main__":
    # load trained model
    gaussian_params = load_model("models/cca_classifier.pkl")

    # test files
    test_files = {
        'reno': [
            '../tcp_flow_capture/traces/parsed/reno_1.csv',
        ],
        'cubic': [
            '../tcp_flow_capture/traces/parsed/cubic_1.csv',
        ],
    }

    # test acc
    confusion_df = test_accuracy(
        csv_files_by_cca=test_files,
        gaussian_params=gaussian_params
    )
