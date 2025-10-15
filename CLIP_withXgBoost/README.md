# CLIP-based Price Prediction Pipeline

A comprehensive solution implementing **Steps 1-3** of the CLIP-based price prediction pipeline for product pricing using multimodal embeddings.

## 🎯 Objective

Predict product prices using a combination of:
- **Text embeddings** from product catalog descriptions  
- **Image embeddings** from product images
- **Normalized embeddings** for optimal ML performance

## 📋 Pipeline Overview

| Step | Description | Status |
|------|-------------|---------|
| 1 | Use CLIP's text encoder for catalog text | ✅ **Implemented** |
| 2 | Use CLIP's image encoder for product images | ✅ **Implemented** |
| 3 | Normalize both embeddings (unit norm) | ✅ **Implemented** |
| 4 | Fuse via concatenation or weighted mean | 🔄 Next Phase |
| 5 | Train multiple regressors (XGBoost, CatBoost, etc.) | 🔄 Next Phase |
| 6 | Combine via stacking or weighted averaging | 🔄 Next Phase |

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Run the setup script
./setup.sh

# Or manually:
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline
```bash
# Activate environment (if using setup.sh)
source venv/bin/activate

# Run full pipeline (download images + generate embeddings)
python main.py
```

### 3. Alternative Execution Options
```bash
# Only download images
python main.py --download-images-only

# Only generate embeddings (if images already downloaded)
python main.py --generate-embeddings-only

# Use different CLIP model
python main.py --model ViT-L/14

# Get help
python main.py --help
```

## 📁 Project Structure

```
CLIP_withXgBoost/
├── dataset/
│   ├── sample_test.csv           # Input data with product info & image URLs
│   └── sample_test_out.csv       # Target prices for training
├── images/                       # Downloaded product images
├── embeddings/                   # Generated embeddings
│   ├── text_embeddings_normalized.npy
│   ├── image_embeddings_normalized.npy
│   ├── metadata.pkl
│   └── embedding_summary.csv
├── outputs/                      # Processing logs
├── main.py                       # Main pipeline orchestrator
├── clip_embeddings.py            # CLIP embedding generation
├── download_images.py            # Image download utility
├── requirements.txt              # Python dependencies
├── setup.sh                      # Environment setup script
└── README.md                     # This file
```

## 🔧 Core Components

### 1. Image Download (`download_images.py`)
- Downloads product images from URLs in CSV
- Handles various image formats (JPG, PNG, WebP, etc.)
- Implements retry logic and error handling
- Validates images and converts to RGB format
- Generates download reports

### 2. CLIP Embeddings (`clip_embeddings.py`)
- **Step 1**: Text encoding using CLIP's text encoder
- **Step 2**: Image encoding using CLIP's image encoder  
- **Step 3**: L2 normalization to unit norm
- Batch processing for memory efficiency
- Supports multiple CLIP models (ViT-B/32, ViT-B/16, ViT-L/14)

### 3. Main Pipeline (`main.py`)
- Orchestrates the complete workflow
- Environment validation
- Dataset verification
- Command-line interface
- Progress tracking and error reporting

## 📊 Data Format

### Input CSV (`dataset/sample_test.csv`)
- **Delimiter**: Pipe (`|`)
- **Required Columns**:
  - `sample_id`: Unique product identifier
  - `catalog_content`: Product description text
  - `image_link`: URL to product image

### Output CSV (`dataset/sample_test_out.csv`)
- **Columns**:
  - `sample_id`: Product identifier
  - `price`: Target price (float)

## 🤖 CLIP Models

| Model | Parameters | Image Resolution | Embedding Dimension |
|-------|------------|------------------|-------------------|
| ViT-B/32 | 151M | 224×224 | 512 |
| ViT-B/16 | 149M | 224×224 | 512 |
| ViT-L/14 | 427M | 224×224 | 768 |

**Default**: ViT-B/32 (good balance of speed and quality)
**Recommended for production**: ViT-L/14 (best quality)

## 📈 Generated Outputs

### 1. Embeddings
- `text_embeddings_normalized.npy`: Normalized text embeddings (N × 512/768)
- `image_embeddings_normalized.npy`: Normalized image embeddings (M × 512/768)

### 2. Metadata
- `metadata.pkl`: Complete metadata including sample IDs, paths, dimensions
- `embedding_summary.csv`: Summary of embedding generation results

### 3. Reports
- `successful_downloads.csv`: Successfully downloaded images
- `failed_downloads.csv`: Failed download attempts

## 🔍 Quality Assurance

### Embedding Normalization Verification
```python
import numpy as np

# Load embeddings
text_emb = np.load('embeddings/text_embeddings_normalized.npy')
image_emb = np.load('embeddings/image_embeddings_normalized.npy')

# Check normalization (should be ~1.0)
text_norms = np.linalg.norm(text_emb, axis=1)
image_norms = np.linalg.norm(image_emb, axis=1)

print(f"Text embedding norms: {text_norms.mean():.6f} ± {text_norms.std():.6f}")
print(f"Image embedding norms: {image_norms.mean():.6f} ± {image_norms.std():.6f}")
```

## ⚙️ Configuration

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: Control GPU usage
- `CLIP_CACHE`: Override CLIP model cache location

### Script Parameters
```python
# In clip_embeddings.py
MODEL_NAME = "ViT-B/32"        # CLIP model
BATCH_SIZE = 32                # Processing batch size
DEVICE = "cuda"                # or "cpu"

# In download_images.py  
MAX_RETRIES = 3                # Download retry attempts
DELAY = 1.0                    # Delay between requests
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use CPU instead
   export CUDA_VISIBLE_DEVICES=""
   python main.py
   ```

2. **Missing Images**
   ```bash
   # Check download logs
   cat outputs/failed_downloads.csv
   
   # Re-run download only
   python main.py --download-images-only
   ```

3. **Import Errors**
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt
   
   # For CLIP specifically
   pip install git+https://github.com/openai/CLIP.git
   ```

### Performance Tips

1. **GPU Acceleration**: Install CUDA-enabled PyTorch for faster processing
2. **Batch Size**: Reduce if encountering memory issues
3. **Model Selection**: Use ViT-B/32 for speed, ViT-L/14 for quality
4. **Parallel Processing**: Modify batch sizes based on your hardware

## 📝 Next Steps (Steps 4-6)

After completing steps 1-3, you'll have normalized embeddings ready for:

### Step 4: Embedding Fusion
```python
# Concatenation approach
combined_emb = np.concatenate([text_emb, image_emb], axis=1)

# Weighted mean approach  
alpha = 0.7  # text weight
combined_emb = alpha * text_emb + (1-alpha) * image_emb
```

### Step 5: Multiple Regressors
```python
# Example with XGBoost and CatBoost
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

models = {
    'xgb': XGBRegressor(),
    'catboost': CatBoostRegressor(verbose=False)
}
```

### Step 6: Model Stacking
```python
# Ensemble predictions
final_prediction = 0.6 * xgb_pred + 0.4 * catboost_pred
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is provided as-is for educational and research purposes.

## 🎉 Acknowledgments

- **OpenAI CLIP** for the foundational multimodal model
- **PyTorch** for deep learning framework
- **Hugging Face** for model distribution and tools

---

**Ready to predict prices with multimodal AI!** 🚀