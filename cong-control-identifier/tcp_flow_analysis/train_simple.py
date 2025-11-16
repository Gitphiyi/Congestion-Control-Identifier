#!/usr/bin/env python3
from train_features import train_classifier, save_model
from globals import (TRAINING_FILES, MODEL_PATH, BANDWIDTH_KBPS,
                     BDP_FACTOR, FEATURE_NUM, MAX_DEG)
import os

"""
SIMPLE TRAINING SCRIPT FOR CCA CLASSIFIER
"""


def main():
    print("="*80)
    print("CCA CLASSIFIER TRAINING")
    print("="*80)

    # training files from globals.py
    csv_files_by_cca = TRAINING_FILES

    print(f"\nTraining Configuration:")
    print(f"  RTT: auto-extracted from CSV files")
    print(f"  Bandwidth: {BANDWIDTH_KBPS} kbps ({BANDWIDTH_KBPS/1000:.1f} Mbit/s)")
    print(f"  BDP Factor: {BDP_FACTOR}")
    print(f"  Feature #: {FEATURE_NUM}")
    print(f"  Max Polynomial Degree: {MAX_DEG}")

    print(f"\nTraining Data:")
    total_files = sum(len(files) for files in csv_files_by_cca.values())
    print(f"  Total CCAs: {len(csv_files_by_cca)}")
    print(f"  Total Files: {total_files}")
    for cca, files in csv_files_by_cca.items():
        print(f"    {cca}: {len(files)} files")

    # train classifier
    print("\nStarting training...\n")
    gaussian_params = train_classifier(
        csv_files_by_cca=csv_files_by_cca
    )

    # save model (path from globals.py)
    save_model(gaussian_params, MODEL_PATH)

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Trained on {total_files} files across {len(gaussian_params)} CCAs")

    return gaussian_params


if __name__ == "__main__":
    main()
