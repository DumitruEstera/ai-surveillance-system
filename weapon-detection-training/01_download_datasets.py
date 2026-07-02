#!/usr/bin/env python3
"""
01_download_datasets.py
Downloads the two weapon detection datasets to /dev/shm/estera/

Dataset 1 (Zenodo): "Dangerous Items" - direct zip download
Dataset 2 (Roboflow): "SOHAS weapon detection" - via roboflow pip package

Usage:
    python 01_download_datasets.py --roboflow-api-key YOUR_KEY

"""

import os
import sys
import argparse
import subprocess
import zipfile
import shutil

BASE_DIR = "/dev/shm/estera"
ZENODO_DIR = os.path.join(BASE_DIR, "zenodo_raw")
ROBOFLOW_DIR = os.path.join(BASE_DIR, "roboflow_raw")


def download_zenodo():
    """Download the Dangerous Items dataset from Zenodo."""
    url = "https://zenodo.org/records/16422779/files/Dangerous%20Items.zip?download=1"
    zip_path = os.path.join(BASE_DIR, "dangerous_items.zip")

    os.makedirs(ZENODO_DIR, exist_ok=True)

    if os.path.exists(ZENODO_DIR) and any(
        f for f in os.listdir(ZENODO_DIR) if not f.startswith(".")
    ):
        print("[Zenodo] Already downloaded, skipping. Delete folder to re-download.")
        return

    print(f"[Zenodo] Downloading (~1.4 GB) ...")
    subprocess.run(
        ["wget", "-c", "-O", zip_path, url],
        check=True,
    )

    print("[Zenodo] Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(ZENODO_DIR)

    os.remove(zip_path)
    print("[Zenodo] Done.")


def download_roboflow(api_key: str):
    """Download the SOHAS weapon detection dataset from Roboflow via REST API."""
    os.makedirs(ROBOFLOW_DIR, exist_ok=True)

    if os.path.exists(ROBOFLOW_DIR) and any(
        f for f in os.listdir(ROBOFLOW_DIR) if not f.startswith(".")
    ):
        print("[Roboflow] Already downloaded, skipping. Delete folder to re-download.")
        return

    import urllib.request
    import json

    # Step 1: Get the download URL from the Roboflow API
    api_url = (
        f"https://api.roboflow.com/aditikulkarni-1710-gmail-com/"
        f"sohas-weapon-detection/2/yolov8?api_key={api_key}"
    )

    print("[Roboflow] Fetching download link ...")
    try:
        with urllib.request.urlopen(api_url) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"[Roboflow] API error: {e}")
        print("[Roboflow] Falling back to manual instructions ...")
        print(f"  1. Go to: https://universe.roboflow.com/aditikulkarni-1710-gmail-com/sohas-weapon-detection/dataset/2")
        print(f"  2. Click 'Download Dataset' -> YOLOv8 format -> download the zip")
        print(f"  3. Extract it to: {ROBOFLOW_DIR}")
        return

    download_url = data.get("export", {}).get("link") or data.get("link")
    if not download_url:
        # Sometimes the response structure differs
        print(f"[Roboflow] API response keys: {list(data.keys())}")
        # Try alternate key
        for key in ["export", "download", "url"]:
            if key in data and isinstance(data[key], dict):
                download_url = data[key].get("link") or data[key].get("url")
                if download_url:
                    break
            elif key in data and isinstance(data[key], str):
                download_url = data[key]
                break

    if not download_url:
        print("[Roboflow] Could not extract download URL from API response.")
        print(f"  Response: {json.dumps(data, indent=2)[:500]}")
        print(f"\n  Manual alternative:")
        print(f"  1. Go to: https://universe.roboflow.com/aditikulkarni-1710-gmail-com/sohas-weapon-detection/dataset/2")
        print(f"  2. Click 'Download Dataset' -> YOLOv8 -> download zip")
        print(f"  3. Extract to: {ROBOFLOW_DIR}")
        return

    # Step 2: Download the zip
    zip_path = os.path.join(BASE_DIR, "roboflow_weapon.zip")
    print(f"[Roboflow] Downloading dataset ...")
    subprocess.run(
        ["wget", "-c", "-O", zip_path, download_url],
        check=True,
    )

    # Step 3: Extract
    print("[Roboflow] Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(ROBOFLOW_DIR)

    os.remove(zip_path)

    # Verify
    contents = os.listdir(ROBOFLOW_DIR)
    print(f"[Roboflow] Done. Contents: {contents}")


def main():
    parser = argparse.ArgumentParser(description="Download weapon detection datasets")
    parser.add_argument(
        "--roboflow-api-key",
        required=True,
        help="Your Roboflow API key",
    )
    args = parser.parse_args()

    os.makedirs(BASE_DIR, exist_ok=True)
    download_zenodo()
    download_roboflow(args.roboflow_api_key)
    print("\n=== All downloads complete ===")
    print(f"Zenodo  -> {ZENODO_DIR}")
    print(f"Roboflow -> {ROBOFLOW_DIR}")


if __name__ == "__main__":
    main()