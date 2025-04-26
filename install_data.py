#!/usr/bin/env python3
"""
Data Installer for ds004496 Dataset

This script downloads and sets up the ds004496 dataset with proper folder structure,
including the derivatives with fMRIPrep data.
Usage: python3 install_data.py
"""

import os
import sys
import argparse
import shutil
import zipfile
from pathlib import Path

def download_with_requests(url, destination):
    """Download a file using the requests library."""
    try:
        import requests
        print(f"Downloading from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1 MB
        
        with open(destination, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    percent = int(100 * downloaded / total_size) if total_size > 0 else 0
                    sys.stdout.write(f"\rDownloading: {percent}% ({downloaded//(1024*1024)} MB)")
                    sys.stdout.flush()
            print()
        return True
    except ImportError:
        print("The requests library is not installed.")
        print("Please install it using: pip install requests")
        return False
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def download_with_urllib(url, destination):
    """Download a file using urllib (standard library)."""
    try:
        import urllib.request
        print(f"Downloading from {url}...")
        
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = int(100 * downloaded / total_size) if total_size > 0 else 0
            sys.stdout.write(f"\rDownloading: {percent}% ({downloaded//(1024*1024)} MB)")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, destination, reporthook=report_progress)
        print()
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def verify_derivatives(dataset_path):
    """Check if the derivatives structure exists and has the required files."""
    expected_paths = [
        "derivatives/fmriprep/sub-01/ses-imagenet01",
        "derivatives/fmriprep/sub-01/ses-imagenet02",
        "derivatives/fmriprep/sub-02/ses-imagenet01",
        "derivatives/fmriprep/sub-02/ses-imagenet02",
    ]
    
    all_exist = True
    for path in expected_paths:
        full_path = os.path.join(dataset_path, path)
        if not os.path.exists(full_path):
            print(f"Missing: {path}")
            all_exist = False
            
    if all_exist:
        print("All expected derivative directories exist.")
        
        for path in expected_paths:
            full_path = os.path.join(dataset_path, path)
            
            # Check for .json, .tsv, and .nii.gz files
            json_files = [f for f in os.listdir(full_path) if f.endswith('.json')]
            tsv_files = [f for f in os.listdir(full_path) if f.endswith('.tsv')]
            nii_files = [f for f in os.listdir(full_path) if f.endswith('.nii.gz')]
            
            print(f"\nChecking files in {path}:")
            print(f"  - JSON files: {len(json_files)} (should be at least 2)")
            print(f"  - TSV files: {len(tsv_files)} (should be at least 1)")
            print(f"  - NII.GZ files: {len(nii_files)} (should be at least 1)")
    
    return all_exist

def download_derivatives(destination):
    """Download the derivatives data."""
    dataset_id = "ds004496"
    derivatives_dir = os.path.join(destination, dataset_id, "derivatives")
    
    os.makedirs(derivatives_dir, exist_ok=True)
    
    derivatives_url = f"https://github.com/OpenNeuroDerivatives/{dataset_id}-fmriprep/archive/refs/heads/main.zip"
    zip_path = os.path.join(destination, f"{dataset_id}-fmriprep.zip")
    
    print(f"Downloading derivatives from {derivatives_url}...")
    
    if not download_with_requests(derivatives_url, zip_path) and not download_with_urllib(derivatives_url, zip_path):
        print("Failed to download derivatives. Will attempt alternate URL...")
        
        derivatives_url = f"https://openneuro.org/crn/derivatives/{dataset_id}/fmriprep/download"
        if not download_with_requests(derivatives_url, zip_path) and not download_with_urllib(derivatives_url, zip_path):
            print("Failed to download derivatives from all attempted sources.")
            return False
    
    try:
        print(f"Extracting derivative files to {derivatives_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            root_dir = zip_ref.namelist()[0].split('/')[0]
            
            zip_ref.extractall(destination)
            
            extract_path = os.path.join(destination, root_dir)
            fmriprep_dir = os.path.join(derivatives_dir, "fmriprep")
            
            os.makedirs(fmriprep_dir, exist_ok=True)
            
            for item in os.listdir(extract_path):
                src = os.path.join(extract_path, item)
                dst = os.path.join(fmriprep_dir, item)
                shutil.move(src, dst)
            
            shutil.rmtree(extract_path, ignore_errors=True)
        
        os.remove(zip_path)
        print("Derivatives successfully installed.")
        return True
    except Exception as e:
        print(f"Error extracting derivative files: {e}")
        return False

def setup_dataset_direct(destination):
    """Download the dataset directly from OpenNeuro without using DataLad."""
    dataset_id = "ds004496"
    dataset_path = os.path.join(destination, dataset_id)
    
    os.makedirs(destination, exist_ok=True)
    
    print(f"Installing dataset {dataset_id} to {dataset_path}")
    
    if os.path.exists(dataset_path):
        print(f"Directory {dataset_path} already exists.")
        response = input("Do you want to remove it and download again? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(dataset_path)
        else:
            has_derivatives = verify_derivatives(dataset_path)
            if not has_derivatives:
                print("The existing dataset is missing derivatives. Will download them...")
                download_derivatives(destination)
            return
    
    url = f"https://openneuro.org/crn/datasets/{dataset_id}/snapshots/1.2.2/download"
    zip_path = os.path.join(destination, f"{dataset_id}.zip")
    
    if not download_with_requests(url, zip_path) and not download_with_urllib(url, zip_path):
        print("Failed to download the dataset.")
        return
    
    try:
        print(f"Extracting files to {dataset_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(destination)
        
        os.remove(zip_path)
        print(f"\nDataset successfully installed to {dataset_path}")
        
        has_derivatives = verify_derivatives(dataset_path)
        if not has_derivatives:
            print("The dataset is missing derivatives. Will download them...")
            download_derivatives(destination)
        
        print("To access the data, navigate to this directory.")
    except Exception as e:
        print(f"Error extracting files: {e}")

def main():
    """Main function to parse arguments and run the installation."""
    parser = argparse.ArgumentParser(description="Install ds004496 dataset with derivatives")
    parser.add_argument("--destination", type=str, default="data",
                       help="Destination directory for the dataset (default: ./data)")
    parser.add_argument("--derivatives-only", action="store_true",
                       help="Download only the derivatives if the main dataset already exists")
    args = parser.parse_args()
    
    if args.derivatives_only:
        dataset_path = os.path.join(args.destination, "ds004496")
        if os.path.exists(dataset_path):
            download_derivatives(args.destination)
        else:
            print(f"Main dataset not found at {dataset_path}. Please download the full dataset first.")
    else:
        setup_dataset_direct(args.destination)

if __name__ == "__main__":
    main()
