#!/usr/bin/env python3
"""
Image Download Script for CLIP-based Price Prediction

This script downloads product images from URLs in the CSV file and saves them locally.
It uses a simple and robust approach with retry logic for failed downloads.
"""

import os
import pandas as pd
import requests
import time
from tqdm import tqdm

# --- Robust Image Download Function ---
def download_image_with_retry(url, filepath, max_retries=3, delay_seconds=5):
    """
    Downloads an image from a URL with a retry mechanism.

    Args:
        url (str): The URL of the image to download.
        filepath (str): The path to save the image file.
        max_retries (int): Maximum number of times to retry the download.
        delay_seconds (int): Seconds to wait between retries.
    
    Returns:
        bool: True if download successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            # Make the request to the URL with a timeout
            response = requests.get(url, stream=True, timeout=15)
            
            # Raise an exception for bad status codes (4xx or 5xx)
            response.raise_for_status()

            # If successful, save the image
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True # Exit the function on success

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt + 1 < max_retries:
                print(f"Retrying in {delay_seconds} seconds...")
                time.sleep(delay_seconds)
            else:
                print(f"All {max_retries} attempts failed for {url}.")
                return False # All retries failed

def download_images_from_csv(csv_path, image_dir="images", delimiter=','):
    """
    Download images from CSV file containing image URLs
    
    Args:
        csv_path (str): Path to CSV file
        image_dir (str): Directory to save images
        delimiter (str): CSV delimiter
    
    Returns:
        tuple: (successful_count, failed_count)
    """
    # Create the directory for images if it doesn't exist
    os.makedirs(image_dir, exist_ok=True)
    
    # Load the CSV file
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, delimiter=delimiter, quotechar='"', skipinitialspace=True)
    
    # Check if required columns exist
    if 'sample_id' not in df.columns or 'image_link' not in df.columns:
        raise ValueError("CSV must contain 'sample_id' and 'image_link' columns")
    
    # Remove rows with missing image links
    initial_count = len(df)
    df = df.dropna(subset=['image_link'])
    final_count = len(df)
    
    if initial_count > final_count:
        print(f"Removed {initial_count - final_count} rows with missing image URLs")
    
    print(f"Found {final_count} products with image URLs")
    print("Starting image download process...")
    
    successful_downloads = []
    failed_downloads = []
    
    # Using tqdm for a progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Downloading Images"):
        sample_id = row['sample_id']
        image_link = row['image_link']
        
        # Create filename using sample_id
        image_name = f"{sample_id}.jpg"  # Use sample_id as filename
        destination_path = os.path.join(image_dir, image_name)

        # Skip if the image already exists
        if not os.path.exists(destination_path):
            success = download_image_with_retry(image_link, destination_path)
            if success:
                successful_downloads.append({
                    'sample_id': sample_id,
                    'image_path': destination_path,
                    'original_url': image_link
                })
            else:
                failed_downloads.append({
                    'sample_id': sample_id,
                    'original_url': image_link
                })
        else:
            # File already exists, count as successful
            successful_downloads.append({
                'sample_id': sample_id,
                'image_path': destination_path,
                'original_url': image_link
            })
    
    print("\nDownload process finished.")
    
    # Save download results
    os.makedirs('outputs', exist_ok=True)
    
    if len(successful_downloads) > 0:
        success_df = pd.DataFrame(successful_downloads)
        success_df.to_csv('outputs/successful_downloads.csv', index=False)
        print(f"Successfully downloaded {len(successful_downloads)} images")
        
    if len(failed_downloads) > 0:
        failed_df = pd.DataFrame(failed_downloads)
        failed_df.to_csv('outputs/failed_downloads.csv', index=False)
        print(f"Failed to download {len(failed_downloads)} images")
    
    # Final Count Verification
    count = len([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"\nTotal images in '{image_dir}': {count}")
    
    print(f"\nDownload Summary:")
    print(f"Total attempted: {len(df)}")
    print(f"Successful: {len(successful_downloads)}")
    print(f"Failed: {len(failed_downloads)}")
    print(f"Success rate: {len(successful_downloads)/len(df)*100:.1f}%")
    
    return len(successful_downloads), len(failed_downloads)

def main():
    """Main function to run image downloading"""
    # Configuration
    CSV_PATH = "dataset/sample_test.csv"
    IMAGE_DIR = "images"
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file {CSV_PATH} not found!")
        return
    
    # Download images using the simple function
    try:
        successful_count, failed_count = download_images_from_csv(CSV_PATH, IMAGE_DIR)
        
        print(f"\nImages saved to: {IMAGE_DIR}")
        print(f"Download logs saved to: outputs/")
        
    except Exception as e:
        print(f"Error during download process: {e}")

if __name__ == "__main__":
    main()