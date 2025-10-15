#!/usr/bin/env python3
"""
CLIP Embeddings Generator for Price Prediction

This script implements Steps 1-3 of the CLIP-based price prediction pipeline:
1. Use CLIP's text encoder for catalog text
2. Use CLIP's image encoder for product images  
3. Normalize both embeddings (unit norm)

The script processes product data and generates normalized embeddings for both
text and images, which can then be used for further ML tasks.
"""

import os
import pandas as pd
import numpy as np
import torch
import clip
from PIL import Image
import pickle
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class CLIPEmbeddingGenerator:
    def __init__(self, model_name="ViT-B/32", device=None):
        """
        Initialize CLIP embedding generator
        
        Args:
            model_name (str): CLIP model to use ('ViT-B/32', 'ViT-B/16', 'ViT-L/14')
            device (str): Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        print(f"Loading CLIP model: {model_name}")
        
        # Load CLIP model and preprocessing
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()  # Set to evaluation mode
        
        print(f"CLIP model loaded successfully!")
        
    def encode_text(self, texts, batch_size=32):
        """
        Step 1: Use CLIP's text encoder for catalog text
        
        Args:
            texts (list): List of text strings to encode
            batch_size (int): Batch size for processing
            
        Returns:
            np.ndarray: Text embeddings of shape (n_texts, embedding_dim)
        """
        print(f"Encoding {len(texts)} text descriptions...")
        
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Text encoding"):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize text
                try:
                    tokens = clip.tokenize(batch_texts, truncate=True).to(self.device)
                    
                    # Get text embeddings
                    batch_embeddings = self.model.encode_text(tokens)
                    
                    # Move to CPU and convert to numpy
                    batch_embeddings = batch_embeddings.cpu().numpy()
                    embeddings.append(batch_embeddings)
                    
                except Exception as e:
                    print(f"Error processing text batch {i//batch_size}: {e}")
                    # Add zero embeddings for failed batch
                    batch_size_actual = len(batch_texts)
                    zero_embeddings = np.zeros((batch_size_actual, 512))  # ViT-B/32 has 512 dims
                    embeddings.append(zero_embeddings)
        
        embeddings = np.vstack(embeddings)
        print(f"Text encoding complete. Shape: {embeddings.shape}")
        
        return embeddings
    
    def encode_images(self, image_paths, batch_size=32):
        """
        Step 2: Use CLIP's image encoder for product images
        
        Args:
            image_paths (list): List of paths to image files
            batch_size (int): Batch size for processing
            
        Returns:
            np.ndarray: Image embeddings of shape (n_images, embedding_dim)
            list: List of successfully processed image paths
        """
        print(f"Encoding {len(image_paths)} images...")
        
        embeddings = []
        valid_paths = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(image_paths), batch_size), desc="Image encoding"):
                batch_paths = image_paths[i:i + batch_size]
                batch_images = []
                batch_valid_paths = []
                
                # Load and preprocess images
                for path in batch_paths:
                    try:
                        if os.path.exists(path):
                            image = Image.open(path).convert('RGB')
                            processed_image = self.preprocess(image)
                            batch_images.append(processed_image)
                            batch_valid_paths.append(path)
                        else:
                            print(f"Warning: Image not found: {path}")
                            
                    except Exception as e:
                        print(f"Error loading image {path}: {e}")
                
                if batch_images:
                    try:
                        # Stack images and move to device
                        batch_tensor = torch.stack(batch_images).to(self.device)
                        
                        # Get image embeddings
                        batch_embeddings = self.model.encode_image(batch_tensor)
                        
                        # Move to CPU and convert to numpy
                        batch_embeddings = batch_embeddings.cpu().numpy()
                        embeddings.append(batch_embeddings)
                        valid_paths.extend(batch_valid_paths)
                        
                    except Exception as e:
                        print(f"Error processing image batch {i//batch_size}: {e}")
        
        if embeddings:
            embeddings = np.vstack(embeddings)
            print(f"Image encoding complete. Shape: {embeddings.shape}")
            print(f"Successfully processed {len(valid_paths)}/{len(image_paths)} images")
        else:
            print("No images were successfully processed!")
            embeddings = np.array([])
            
        return embeddings, valid_paths
    
    def normalize_embeddings(self, embeddings):
        """
        Step 3: Normalize embeddings to unit norm
        
        Args:
            embeddings (np.ndarray): Embeddings to normalize
            
        Returns:
            np.ndarray: L2-normalized embeddings
        """
        if len(embeddings) == 0:
            return embeddings
            
        print(f"Normalizing embeddings of shape: {embeddings.shape}")
        
        # L2 normalization (unit norm)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        
        normalized_embeddings = embeddings / norms
        
        # Verify normalization
        sample_norms = np.linalg.norm(normalized_embeddings[:5], axis=1)
        print(f"Sample norms after normalization: {sample_norms}")
        
        return normalized_embeddings
    
    def process_dataset(self, csv_path, image_dir="images", save_embeddings=True):
        """
        Process the entire dataset and generate embeddings
        
        Args:
            csv_path (str): Path to CSV file
            image_dir (str): Directory containing images
            save_embeddings (bool): Whether to save embeddings to disk
            
        Returns:
            dict: Dictionary containing all processed data
        """
        print(f"Processing dataset: {csv_path}")
        
        # Load data
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} samples")
        
        # Prepare text data
        print("\n=== Step 1: Text Encoding ===")
        catalog_texts = []
        
        for idx, row in df.iterrows():
            # Combine all text fields for richer representation
            text_parts = []
            
            if 'catalog_content' in df.columns and pd.notna(row['catalog_content']):
                text_parts.append(str(row['catalog_content']))
            
            # Combine all text into one string
            full_text = ' '.join(text_parts) if text_parts else "No description available"
            catalog_texts.append(full_text)
        
        # Generate text embeddings
        text_embeddings = self.encode_text(catalog_texts)
        
        # Normalize text embeddings
        text_embeddings_normalized = self.normalize_embeddings(text_embeddings)
        
        # Prepare image data
        print("\n=== Step 2: Image Encoding ===")
        image_paths = []
        valid_sample_ids = []
        
        for idx, row in df.iterrows():
            sample_id = row['sample_id']
            
            # Look for image files with common extensions
            for ext in ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp']:
                image_path = os.path.join(image_dir, f"{sample_id}.{ext}")
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                    valid_sample_ids.append(sample_id)
                    break
            else:
                print(f"Warning: No image found for sample_id {sample_id}")
        
        print(f"Found {len(image_paths)} images for {len(df)} samples")
        
        # Generate image embeddings
        if image_paths:
            image_embeddings, valid_image_paths = self.encode_images(image_paths)
            
            if len(image_embeddings) > 0:
                # Normalize image embeddings
                print("\n=== Step 3: Embedding Normalization ===")
                image_embeddings_normalized = self.normalize_embeddings(image_embeddings)
            else:
                image_embeddings_normalized = np.array([])
        else:
            image_embeddings_normalized = np.array([])
            valid_image_paths = []
        
        # Prepare results
        results = {
            'text_embeddings': text_embeddings_normalized,
            'image_embeddings': image_embeddings_normalized,
            'sample_ids': df['sample_id'].tolist(),
            'valid_image_sample_ids': valid_sample_ids[:len(image_embeddings_normalized)] if len(image_embeddings_normalized) > 0 else [],
            'catalog_texts': catalog_texts,
            'image_paths': valid_image_paths,
            'embedding_dimension': text_embeddings_normalized.shape[1] if len(text_embeddings_normalized) > 0 else 0,
            'model_name': self.model_name
        }
        
        if save_embeddings:
            print("\n=== Saving Results ===")
            
            # Save embeddings
            embeddings_dir = Path('embeddings')
            embeddings_dir.mkdir(exist_ok=True)
            
            # Save as numpy arrays
            if len(text_embeddings_normalized) > 0:
                np.save(embeddings_dir / 'text_embeddings_normalized.npy', text_embeddings_normalized)
                print(f"Text embeddings saved: {text_embeddings_normalized.shape}")
            
            if len(image_embeddings_normalized) > 0:
                np.save(embeddings_dir / 'image_embeddings_normalized.npy', image_embeddings_normalized)
                print(f"Image embeddings saved: {image_embeddings_normalized.shape}")
            
            # Save metadata
            metadata = {
                'sample_ids': results['sample_ids'],
                'valid_image_sample_ids': results['valid_image_sample_ids'],
                'catalog_texts': results['catalog_texts'],
                'image_paths': results['image_paths'],
                'embedding_dimension': results['embedding_dimension'],
                'model_name': results['model_name'],
                'text_embeddings_shape': text_embeddings_normalized.shape if len(text_embeddings_normalized) > 0 else (0, 0),
                'image_embeddings_shape': image_embeddings_normalized.shape if len(image_embeddings_normalized) > 0 else (0, 0)
            }
            
            with open(embeddings_dir / 'metadata.pkl', 'wb') as f:
                pickle.dump(metadata, f)
                
            print(f"Metadata saved to embeddings/metadata.pkl")
            
            # Save summary CSV
            summary_data = []
            for i, sample_id in enumerate(results['sample_ids']):
                row_data = {
                    'sample_id': sample_id,
                    'has_text_embedding': i < len(text_embeddings_normalized),
                    'has_image_embedding': sample_id in results['valid_image_sample_ids'],
                    'text_preview': results['catalog_texts'][i][:100] + "..." if len(results['catalog_texts'][i]) > 100 else results['catalog_texts'][i]
                }
                summary_data.append(row_data)
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(embeddings_dir / 'embedding_summary.csv', index=False)
            print(f"Summary saved to embeddings/embedding_summary.csv")
        
        print("\n=== Processing Complete ===")
        print(f"Text embeddings: {text_embeddings_normalized.shape if len(text_embeddings_normalized) > 0 else 'None'}")
        print(f"Image embeddings: {image_embeddings_normalized.shape if len(image_embeddings_normalized) > 0 else 'None'}")
        
        return results

def main():
    """Main function to run CLIP embedding generation"""
    
    # Configuration
    CSV_PATH = "dataset/sample_test.csv"
    IMAGE_DIR = "images"
    MODEL_NAME = "ViT-B/32"  # You can change to "ViT-B/16" or "ViT-L/14" for better quality
    
    # Check if CSV exists
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file {CSV_PATH} not found!")
        print("Please make sure the dataset file exists.")
        return
    
    try:
        # Create CLIP embedding generator
        generator = CLIPEmbeddingGenerator(model_name=MODEL_NAME)
        
        # Process dataset
        results = generator.process_dataset(CSV_PATH, IMAGE_DIR)
        
        print("\n" + "="*60)
        print("CLIP EMBEDDING GENERATION COMPLETE!")
        print("="*60)
        print(f"✓ Step 1: Text embeddings generated and normalized")
        print(f"✓ Step 2: Image embeddings generated and normalized")  
        print(f"✓ Step 3: All embeddings normalized to unit norm")
        print("\nNext steps:")
        print("4. Fuse embeddings via concatenation or weighted mean")
        print("5. Train multiple regressors (XGBoost, CatBoost, etc.)")
        print("6. Combine via stacking or weighted averaging")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()