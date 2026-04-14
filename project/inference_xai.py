"""
Simple inference script with GRAD-CAM and LIME for a single image
"""
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
IMAGE_PATH = r"C:\Users\maila\Desktop\MDM_Defect_Classification\combined_all\2885.jpg"
MODEL_PATH = r"C:\Users\maila\Desktop\MDM_Defect_Classification\project\resnet18_transfer_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224

# Class labels (binary classification: 0=OK, 1=Not OK)
CLASS_NAMES = {0: 'OK', 1: 'Not OK (Defect)'}

print(f"Device: {DEVICE}")

# ==================== LOAD MODEL ====================
def load_model():
    model = models.resnet18(pretrained=False)
    # ResNet18 has 512 features in final layer
    model.fc = nn.Linear(512, 2)  # Binary classification
    
    # Load weights
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(DEVICE)
    model.eval()
    return model

# ==================== IMAGE PREPROCESSING ====================
def load_and_preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image, image_tensor

# ==================== GRAD-CAM ====================
class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer
        self.gradients = None
        self.activations = None
        
        layer.register_forward_hook(self.save_activation)
        layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output[0, class_idx]
        target.backward()
        
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        weights = gradients.mean(dim=(1, 2))  # [C]
        cam = torch.sum(weights.view(-1, 1, 1) * activations, dim=0)  # [H, W]
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()

# ==================== LIME ====================
try:
    import lime
    import lime.lime_image
    
    def lime_explain(image_array, model, device):
        """Simple LIME explanation"""
        def predict_fn(images):
            # images is a numpy array of shape (N, H, W, 3)
            transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            predictions = []
            with torch.no_grad():
                for img in images:
                    img_pil = Image.fromarray(img.astype('uint8'))
                    img_tensor = transform(img_pil).unsqueeze(0).to(device)
                    output = model(img_tensor)
                    probs = torch.softmax(output, dim=1)
                    predictions.append(probs.cpu().numpy())
            
            return np.vstack(predictions)
        
        explainer = lime.lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image_array, 
            predict_fn, 
            top_labels=2,
            hide_color=0,
            num_samples=100
        )
        
        return explanation
    
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available")

# ==================== MAIN INFERENCE ====================
def main():
    print("\n" + "="*60)
    print("INFERENCE & XAI ANALYSIS")
    print("="*60)
    
    # Load model
    print("\n[1/4] Loading model...")
    model = load_model()
    print("✓ Model loaded successfully")
    
    # Load and preprocess image
    print(f"\n[2/4] Loading image from: {IMAGE_PATH}")
    if not os.path.exists(IMAGE_PATH):
        print(f"✗ Image not found: {IMAGE_PATH}")
        return
    
    original_image, image_tensor = load_and_preprocess_image(IMAGE_PATH)
    image_array = np.array(original_image)
    print(f"✓ Image loaded: {image_array.shape}")
    
    # INFERENCE
    print("\n[3/4] Running inference...")
    with torch.no_grad():
        input_batch = image_tensor.unsqueeze(0).to(DEVICE)
        output = model(input_batch)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[predicted_class].item()
    
    print(f"✓ Prediction: {CLASS_NAMES[predicted_class]} (Confidence: {confidence:.2%})")
    print(f"  - OK probability: {probabilities[0].item():.2%}")
    print(f"  - Defect probability: {probabilities[1].item():.2%}")
    
    # GRAD-CAM
    print("\n[4/4] Generating XAI visualizations...")
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate(input_batch, predicted_class)
    print(f"✓ GRAD-CAM generated")
    
    # LIME
    if LIME_AVAILABLE:
        try:
            explanation = lime_explain(image_array, model, DEVICE)
            print(f"✓ LIME explanation generated")
            lime_available = True
        except Exception as e:
            print(f"⚠ LIME failed: {str(e)}")
            lime_available = False
    else:
        lime_available = False
    
    # ==================== VISUALIZATION ====================
    print("\nGenerating visualization...")
    
    if lime_available:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # GRAD-CAM overlay
    cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam_colored = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
    
    overlay = Image.blend(original_image, Image.fromarray(cam_colored), alpha=0.5)
    axes[1].imshow(overlay)
    axes[1].set_title('GRAD-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # LIME
    if lime_available:
        image_and_mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        axes[2].imshow(image_and_mask[0])
        axes[2].set_title('LIME Top Features', fontsize=12, fontweight='bold')
        axes[2].axis('off')
    
    # Add title
    title = f"Classification: {CLASS_NAMES[predicted_class]} ({confidence:.1%})"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    output_path = r"C:\Users\maila\Desktop\MDM_Defect_Classification\project\inference_result.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_path}")
    plt.show()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
