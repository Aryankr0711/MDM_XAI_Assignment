# XAI Industrial Image Defect Classification - Setup and Usage Guide

## Project Overview

This project implements a comprehensive **Explainable AI (XAI)** pipeline for industrial image classification. It demonstrates how to build a trustworthy deep learning model with interpretability at its core, combining:

- **CNN Models:** Custom CNN and ResNet18 transfer learning
- **XAI Techniques:** SHAP (global), Grad-CAM (local), LIME (local interpretable model-agnostic)
- **Complete Pipeline:** From data preparation to comprehensive report generation

## Directory Structure

```
MDM_Defect_Classification/
├── Dataset/                          # Original dataset (~5200 images)
│   ├── Renamed_OK/                   # Class 1: Defect-free objects
│   └── Renamed_Not_OK/               # Class 0: Defective objects
├── combined_all/                     # Sampled subset (1000 images)
└── project/                          # Main project directory
    ├── XAI_Industrial_Classification.ipynb    # Main notebook
    ├── XAI_Analysis_Report.md                 # Generated report
    ├── requirements.txt                       # Dependencies
    ├── image_labels.csv                       # Full label mapping
    ├── train_labels.csv                       # Training set
    ├── val_labels.csv                         # Validation set
    ├── test_labels.csv                        # Test set
    ├── custom_cnn_model.pth                   # Trained custom CNN
    ├── resnet18_transfer_model.pth            # Trained ResNet18
    └── [visualization PNGs]                   # Generated plots
```

## Installation and Setup

### Step 1: Create Python Environment

```bash
# Navigate to project directory
cd C:\Users\maila\Desktop\MDM_Defect_Classification\project

# Create virtual environment (recommended)
python -m venv xai_env

# Activate environment
# On Windows:
xai_env\Scripts\activate
# On macOS/Linux:
source xai_env/bin/activate
```

### Step 2: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### Step 3: Verify Dataset

Ensure your dataset structure is correct:

```
C:\Users\maila\Desktop\MDM_Defect_Classification\
├── Dataset/
│   ├── Renamed_OK/           (contains .jpg, .png, etc.)
│   └── Renamed_Not_OK/       (contains .jpg, .png, etc.)
```

Run this to check:
```python
from pathlib import Path
dataset_dir = Path(r"C:\Users\maila\Desktop\MDM_Defect_Classification\Dataset")
print(f"OK images: {len(list((dataset_dir / 'Renamed_OK').glob('*')))}")
print(f"Not OK images: {len(list((dataset_dir / 'Renamed_Not_OK').glob('*')))}")
```

## Running the Notebook

### Option 1: Using Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open the notebook file
# Navigate to the project folder and open: XAI_Industrial_Classification.ipynb
```

### Option 2: Using JupyterLab

```bash
jupyter lab
```

### Option 3: Using VS Code

1. Open VS Code
2. Install "Jupyter" extension by Microsoft
3. Open `XAI_Industrial_Classification.ipynb`
4. Select Python kernel from environment
5. Run cells sequentially

## Notebook Workflow

The notebook consists of 12 main sections:

### Section 1: Dataset Preparation (10-15 min)
- Randomly samples 500 images from each class
- Copies to `combined_all` directory
- Fixed seed (42) for reproducibility

### Section 2: Create CSV Mapping (1-2 min)
- Generates `image_labels.csv` with columns: `image_name`, `label`
- Label mapping: 1 = OK (defect-free), 0 = Not OK (defective)

### Section 3: Image Preprocessing (5-10 min)
- Resizes images to 224×224 pixels
- Applies ImageNet normalization
- Defines augmentation strategies for training
- Visualizes preprocessing examples

### Section 4: Train-Val-Test Split (5 min)
- 70% training, 15% validation, 15% test
- Stratified split to maintain class balance
- Creates PyTorch DataLoaders with batch processing

### Section 5: Build Custom CNN (15-30 min)
- Implements custom 3-block CNN architecture
- Binary cross-entropy loss + Adam optimizer
- Trains for 20 epochs
- Saves model weights

### Section 6: Transfer Learning (20-40 min)
- Loads pretrained ResNet18
- Fine-tunes last layers only
- Lower learning rate for stability
- **Main model used for XAI analysis**

### Section 7: Model Evaluation (5 min)
- Evaluates on test set
- Computes: Accuracy, Precision, Recall, F1-Score
- Generates confusion matrix and training curves

### Section 8: SHAP Global Explanations (10-15 min)
- GradientExplainer for pixel importance
- Generates SHAP summary plots
- Creates SHAP heatmaps overlaid on images
- Identifies globally important regions

### Section 9: Grad-CAM Local Explanations (5-10 min)
- Implements Grad-CAM from scratch
- Generates per-sample heatmaps
- Shows decision regions for individual predictions
- Faster than SHAP, good for real-time applications

### Section 10: LIME Local Explanations (10-15 min)
- Model-agnostic interpretability
- Super-pixel based explanations
- Complements Grad-CAM findings
- Validates consistency across methods

### Section 11: Comprehensive Analysis (10-15 min)
- Compares multiple XAI methods side-by-side
- Analyzes model behavior and biases
- Identifies failure cases
- Generates recommendations

### Section 12: Generate Report (2-3 min)
- Creates comprehensive Markdown report
- Includes all findings and visualizations
- Ready for academic submission
- Saved as `XAI_Analysis_Report.md`

## Expected Runtime

- **Total Execution Time:** 90-150 minutes (depending on hardware)
- **CPU-only:** ~3-4 hours
- **GPU-enabled:** ~45-60 minutes
- **Data preparation:** ~20 minutes
- **Model training:** ~40-80 minutes (largest portion)
- **XAI analysis:** ~30-40 minutes

### Hardware Recommendations

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| RAM | 8 GB | 16 GB | 32 GB |
| Storage | 5 GB | 10 GB | 20 GB |
| GPU | None | NVIDIA GTX 1060 | NVIDIA RTX 3080+ |
| Processor | i5 | i7 | Threadripper |

## Output Files

After running the complete notebook, you'll have:

### Data Files
- `image_labels.csv` - Complete dataset mapping (1000 images)
- `train_labels.csv` - 700 training images
- `val_labels.csv` - 150 validation images
- `test_labels.csv` - 150 test images

### Model Files
- `custom_cnn_model.pth` - Custom CNN weights
- `resnet18_transfer_model.pth` - Fine-tuned ResNet18 weights

### Visualization Files
- `preprocessing_examples.png` - Sample preprocessed images
- `training_curves.png` - Loss and accuracy curves over epochs
- `confusion_matrix.png` - Test set confusion matrix
- `shap_summary.png` - SHAP feature importance bar plot
- `shap_heatmaps.png` - SHAP heatmaps on sample images
- `gradcam_visualizations.png` - Grad-CAM attention maps (9 samples)
- `lime_explanations.png` - LIME explanations (9 samples)
- `xai_comprehensive_analysis.png` - Combined comparison of all methods

### Report
- `XAI_Analysis_Report.md` - Comprehensive analysis report (~15 pages)

## Customization Guide

### Modify Dataset Parameters

```python
# In notebook cell 1, add after setup:
SAMPLES_PER_CLASS = 750  # Change from 500 to 750
RANDOM_SEED = 42  # For reproducibility
IMG_SIZE = 256  # Change from 224 to 256
```

### Adjust Model Parameters

```python
# Training parameters
NUM_EPOCHS = 30  # Change from 20
BATCH_SIZE = 64  # Change from 32
LEARNING_RATE = 0.0005  # Adjust learning rate
```

### Use Different Pretrained Models

```python
# In the transfer learning section:
# Option 1: ResNet50 (larger, more powerful)
model = models.resnet50(pretrained=True)

# Option 2: EfficientNet
model = models.efficientnet_b0(pretrained=True)

# Option 3: VGG16
model = models.vgg16(pretrained=True)
```

## Troubleshooting

### Issue: Dataset Not Found
```
Error: Dataset directory not found
Solution: Verify path matches your system:
- Check: C:\Users\maila\Desktop\MDM_Defect_Classification\Dataset exists
- Adjust paths in cell 1 if different
```

### Issue: CUDA Out of Memory
```
Error: CUDA out of memory
Solutions:
1. Reduce BATCH_SIZE from 32 to 16
2. Use CPU: Set device to 'cpu' in cell 1
3. Reduce image size from 224 to 192
4. Use Google Colab (free GPU)
```

### Issue: Module Not Found
```
Error: No module named 'shap'
Solution: Install missing package:
pip install shap lime opencv-python
```

### Issue: Slow Execution
```
Solutions:
1. Use GPU instead of CPU
2. Reduce NUM_EPOCHS to 10
3. Use smaller SAMPLES_PER_CLASS
4. Close other applications
```

## Using Results for Academic Submission

### For Research Papers

1. **Include Visualizations:**
   - Add training curves (Figure 1)
   - Include confusion matrix (Figure 2)
   - Show Grad-CAM examples (Figure 3)
   - Display SHAP analysis (Figure 4)

2. **Write Methodology Section:**
   - Reference the notebook cells
   - Cite PyTorch, SHAP, Grad-CAM papers
   - Include model architecture details

3. **Results Section:**
   - Report metrics from evaluation
   - Discuss XAI findings
   - Address biases and limitations

### For Assignments/Reports

1. **Structure Report:**
   - Copy content from `XAI_Analysis_Report.md`
   - Add required sections (abstract, conclusion)
   - Include all visualizations

2. **Add Interpretations:**
   - Explain why each XAI method matters
   - Discuss findings for each method
   - Connect to domain knowledge

3. **Document Code:**
   - Reference notebook sections
   - Explain key functions
   - Justify architectural choices

## Advanced Topics

### Fine-tuning Strategies

```python
# Selective layer unfreezing
for name, param in model.named_parameters():
    if 'layer4' in name:
        param.requires_grad = True  # Unfreeze layer4
    else:
        param.requires_grad = False  # Freeze others
```

### Custom Loss Functions

```python
# Weighted loss for handling class imbalance
weights = torch.tensor([1.0, 2.0])  # Penalize minority class
criterion = nn.BCEWithLogitsLoss(pos_weight=weights[1])
```

### Ensemble Methods

```python
# Combine multiple models
predictions = (
    model1_pred * 0.4 +
    model2_pred * 0.3 +
    model3_pred * 0.3
)
```

## Citation and References

If using this project, cite:

```bibtex
@misc{bianchi2024xai,
  title={Explainable AI for Industrial Image Defect Classification},
  author={Your Name},
  year={2024},
  note={XAI Implementation with SHAP and Grad-CAM}
}
```

### Key Papers Referenced

1. **SHAP:** Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions.
2. **Grad-CAM:** Selvaraju, R. R., et al. (2016). Grad-CAM: Visual explanations from deep networks.
3. **LIME:** Ribeiro, M. T., et al. (2016). "Why should I trust you?" Explaining predictions.

## Support and Questions

1. **Check Troubleshooting section** above
2. **Review cell comments** in the notebook
3. **Verify dataset structure** matches requirements
4. **Test on smaller subset** first (e.g., 100 images)
5. **Use print statements** to debug

## License and Acknowledgments

This implementation combines:
- PyTorch for deep learning
- SHAP for global explanations
- Grad-CAM for attention visualizations
- LIME for model-agnostic interpretability

References and implementations adapted from official documentation and research papers.

---

**Version:** 1.0  
**Last Updated:** April 13, 2026  
**Status:** Ready for academic submission
