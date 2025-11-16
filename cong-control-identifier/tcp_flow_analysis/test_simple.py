#!/usr/bin/env python3
from test_features import classify_trace, test_accuracy, load_model
from globals import MODEL_PATH, TESTING_FILES
import os

"""
SIMPLE TESTING SCRIPT FOR CCA CLASSIFIER
"""

def classify_single_trace(csv_path, model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        print(f"\nError: Model not found at {model_path}")
        return

    gaussian_params = load_model(model_path)

    print(f"\nClassifying: {csv_path}")
    print(f"Model: {model_path}")
    print(f"Trained CCAs: {list(gaussian_params.keys())}\n")

    # classify
    result = classify_trace(
        csv_path=csv_path,
        gaussian_params=gaussian_params
    )

    print("="*80)
    print("CLASSIFICATION RESULTS")
    print("="*80)
    print(f"\nPredicted CCA: {result['predicted_cca']}")

    print("\nTop 3 Predictions:")
    for i, (cca, raw_prob, norm_prob) in enumerate(result['top3'], 1):
        bar = "█" * int(norm_prob * 50)
        print(f"  {i}. {cca:10s} {bar} {norm_prob:.4f}")

    print("\nRaw Probabilities:")
    for cca, prob in result['probabilities'].items():
        print(f"  {cca}: {prob:.6e}")

    return result


def test_multiple_traces(model_path=MODEL_PATH):
    # test accuracy on multiple files and show confusion matrix

    # load trained model
    if not os.path.exists(model_path):
        print(f"\nError: Model not found at {model_path}")
        return

    gaussian_params = load_model(model_path)

    test_files = TESTING_FILES

    # test accuracy
    confusion_df = test_accuracy(
        csv_files_by_cca=test_files,
        gaussian_params=gaussian_params
    )

    return confusion_df


def main():
    """
    Main test function - choose what to do
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_simple.py <csv_file>           # Classify single file")
        print("  python test_simple.py --accuracy           # Test accuracy on multiple files")
        print("\nExample:")
        print("  python test_simple.py tcp_flow_capture/traces/parsed/reno_2.csv")
        return

    if sys.argv[1] == "--accuracy":
        test_multiple_traces()
    else:
        csv_path = sys.argv[1]
        classify_single_trace(csv_path)


if __name__ == "__main__":
    main()
