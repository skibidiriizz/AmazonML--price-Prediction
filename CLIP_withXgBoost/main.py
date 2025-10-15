#!/usr/bin/env python3
"""
Main Processing Script for CLIP-based Price Prediction Pipeline

This script orchestrates the complete workflow for Steps 1-3:
1. Download images from URLs in the CSV file
2. Generate CLIP text embeddings from catalog content 
3. Generate CLIP image embeddings from product images
4. Normalize both embeddings to unit norm

Usage:
    python main.py [--download-images] [--generate-embeddings] [--model ViT-B/32]
"""

import argparse
import os
import sys
from pathlib import Path

def setup_environment():
    """Check if required packages are installed"""
    print("Checking environment setup...")
    
    missing_packages = []
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError:
        missing_packages.append('torch')
    
    try:
        import clip
        print("✓ CLIP: Available")
    except ImportError:
        missing_packages.append('clip (git+https://github.com/openai/CLIP.git)')
    
    try:
        import pandas as pd
        print(f"✓ Pandas: {pd.__version__}")
    except ImportError:
        missing_packages.append('pandas')
    
    try:
        from PIL import Image
        print("✓ Pillow: Available")
    except ImportError:
        missing_packages.append('Pillow')
    
    try:
        import requests
        print("✓ Requests: Available")
    except ImportError:
        missing_packages.append('requests')
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install required packages:")
        print("pip install -r requirements.txt")
        return False
    
    print("✓ All required packages are available!")
    return True

def download_images():
    """Run image download process"""
    print("\n" + "="*60)
    print("STEP: DOWNLOADING IMAGES")
    print("="*60)
    
    try:
        from download_images import download_images_from_csv
        
        CSV_PATH = "dataset/sample_test.csv"
        IMAGE_DIR = "images"
        
        if not os.path.exists(CSV_PATH):
            print(f"❌ Error: CSV file {CSV_PATH} not found!")
            return False
        
        successful_count, failed_count = download_images_from_csv(CSV_PATH, IMAGE_DIR)
        
        if successful_count > 0:
            print(f"✓ Successfully downloaded {successful_count} images")
            return True
        else:
            print("❌ No images were downloaded successfully")
            return False
            
    except Exception as e:
        print(f"❌ Error during image download: {e}")
        return False

def generate_embeddings(model_name="ViT-B/32"):
    """Run CLIP embedding generation"""
    print("\n" + "="*60)
    print("STEPS 1-3: GENERATING CLIP EMBEDDINGS")
    print("="*60)
    
    try:
        from clip_embeddings import CLIPEmbeddingGenerator
        
        CSV_PATH = "dataset/sample_test.csv"
        IMAGE_DIR = "images"
        
        if not os.path.exists(CSV_PATH):
            print(f"❌ Error: CSV file {CSV_PATH} not found!")
            return False
        
        # Create CLIP embedding generator
        generator = CLIPEmbeddingGenerator(model_name=model_name)
        
        # Process dataset
        results = generator.process_dataset(CSV_PATH, IMAGE_DIR)
        
        # Check results
        text_success = len(results['text_embeddings']) > 0
        image_success = len(results['image_embeddings']) > 0
        
        if text_success and image_success:
            print("✓ Both text and image embeddings generated successfully!")
            return True
        elif text_success:
            print("⚠️  Text embeddings generated, but no image embeddings (missing images?)")
            return True
        elif image_success:
            print("⚠️  Image embeddings generated, but no text embeddings")
            return True
        else:
            print("❌ No embeddings were generated successfully")
            return False
            
    except Exception as e:
        print(f"❌ Error during embedding generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_dataset():
    """Check if dataset files exist"""
    print("\n" + "="*60)
    print("CHECKING DATASET")
    print("="*60)
    
    CSV_PATH = "dataset/sample_test.csv"
    OUTPUT_PATH = "dataset/sample_test_out.csv"
    
    if not os.path.exists(CSV_PATH):
        print(f"❌ Input CSV not found: {CSV_PATH}")
        return False
    
    if not os.path.exists(OUTPUT_PATH):
        print(f"⚠️  Output CSV not found: {OUTPUT_PATH} (needed for training later)")
    else:
        print(f"✓ Output CSV found: {OUTPUT_PATH}")
    
    # Check CSV content
    try:
        import pandas as pd
        df = pd.read_csv(CSV_PATH)
        print(f"✓ Dataset loaded: {len(df)} samples")
        
        required_cols = ['sample_id', 'catalog_content', 'image_link']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        print(f"✓ All required columns present: {required_cols}")
        
        # Check for missing data
        missing_text = df['catalog_content'].isna().sum()
        missing_images = df['image_link'].isna().sum()
        
        print(f"📊 Data completeness:")
        print(f"   - Missing text: {missing_text}/{len(df)} ({missing_text/len(df)*100:.1f}%)")
        print(f"   - Missing image URLs: {missing_images}/{len(df)} ({missing_images/len(df)*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking dataset: {e}")
        return False

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='CLIP-based Price Prediction Pipeline (Steps 1-3)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Full pipeline (download + embeddings)
  python main.py --download-images-only       # Only download images
  python main.py --generate-embeddings-only   # Only generate embeddings
  python main.py --model ViT-L/14             # Use larger CLIP model
        """
    )
    
    parser.add_argument('--download-images-only', action='store_true',
                       help='Only download images, skip embedding generation')
    parser.add_argument('--generate-embeddings-only', action='store_true', 
                       help='Only generate embeddings, skip image download')
    parser.add_argument('--model', default='ViT-B/32',
                       choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
                       help='CLIP model to use (default: ViT-B/32)')
    parser.add_argument('--skip-env-check', action='store_true',
                       help='Skip environment setup check')
    
    args = parser.parse_args()
    
    print("🚀 CLIP-based Price Prediction Pipeline")
    print("Implementing Steps 1-3: Text Encoding, Image Encoding, Normalization")
    print("-" * 60)
    
    # Environment check
    if not args.skip_env_check:
        if not setup_environment():
            print("\n❌ Environment setup failed. Please install required packages.")
            sys.exit(1)
    
    # Dataset check
    if not check_dataset():
        print("\n❌ Dataset check failed. Please verify your data files.")
        sys.exit(1)
    
    # Determine what to run
    download_imgs = not args.generate_embeddings_only
    generate_embs = not args.download_images_only
    
    print(f"\n📋 Execution plan:")
    print(f"   - Download images: {'✓' if download_imgs else '✗'}")
    print(f"   - Generate embeddings: {'✓' if generate_embs else '✗'}")
    if generate_embs:
        print(f"   - CLIP model: {args.model}")
    
    # Execute pipeline
    success = True
    
    if download_imgs:
        if not download_images():
            print("❌ Image download failed")
            success = False
        else:
            print("✓ Image download completed")
    
    if generate_embs and success:
        if not generate_embeddings(args.model):
            print("❌ Embedding generation failed")
            success = False
        else:
            print("✓ Embedding generation completed")
    
    # Final summary
    print("\n" + "="*60)
    if success:
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("✓ Step 1: CLIP text encoder applied to catalog content")
        print("✓ Step 2: CLIP image encoder applied to product images")
        print("✓ Step 3: Both embeddings normalized to unit norm")
        
        print("\n📁 Generated files:")
        if os.path.exists("embeddings/text_embeddings_normalized.npy"):
            print("   - embeddings/text_embeddings_normalized.npy")
        if os.path.exists("embeddings/image_embeddings_normalized.npy"):
            print("   - embeddings/image_embeddings_normalized.npy")
        if os.path.exists("embeddings/metadata.pkl"):
            print("   - embeddings/metadata.pkl")
        if os.path.exists("embeddings/embedding_summary.csv"):
            print("   - embeddings/embedding_summary.csv")
        
        print(f"\n📦 Images downloaded to: images/")
        print(f"📊 Embeddings saved to: embeddings/")
        
        print("\n🚀 Next Steps (Steps 4-6):")
        print("4. Fuse embeddings via concatenation or weighted mean")
        print("5. Train multiple regressors (XGBoost, CatBoost, etc.)")
        print("6. Combine predictions via stacking or weighted averaging")
        
    else:
        print("❌ PIPELINE FAILED!")
        print("="*60)
        print("Please check the error messages above and resolve any issues.")
        sys.exit(1)

if __name__ == "__main__":
    main()