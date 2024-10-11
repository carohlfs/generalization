import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the image
image_path = 'lionhabitat.png'  # Change this to your image path as needed
image = Image.open(image_path).convert('RGB')

# Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

# Ensure input tensor requires gradients
input_batch.requires_grad = True

# Load the pretrained MobileNet v3 large model
model = mobilenet_v3_large(pretrained=True)
model.eval()

# Implement Layer-wise Relevance Propagation (LRP)
def lrp(model, input_tensor, index=None):
    output = model(input_tensor)
    if index is None:
        index = output.argmax(dim=1).item()
    one_hot = torch.zeros_like(output)
    one_hot[0, index] = 1
    output.backward(gradient=one_hot, retain_graph=True)
    
    relevance = input_tensor.grad.data.abs()
    relevance = relevance.squeeze().cpu().numpy()
    
    return relevance

# Compute LRP for the given image
relevance_scores = lrp(model, input_batch)

# Check the shape and ensure it's correct
print(f'Original relevance_scores shape: {relevance_scores.shape}')

# If relevance_scores has three dimensions, average over the color channels
if relevance_scores.ndim == 3:
    relevance_scores = np.mean(relevance_scores, axis=0)

# Normalize the relevance scores for visualization
relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min())

# Enhance the contrast of the heatmap
relevance_scores = np.uint8(255 * relevance_scores)
relevance_scores = cv2.equalizeHist(relevance_scores)

# Create a heatmap and save the image
plt.imshow(relevance_scores, cmap='hot', interpolation='nearest')
plt.axis('off')
plt.colorbar()
heatmap_path = 'lionhabitat_lrp_heatmap.png'
plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
plt.close()

# Display the saved heatmap
saved_heatmap = Image.open(heatmap_path)
saved_heatmap.show()

# Superimpose the heatmap on the original image
heatmap_img = cv2.applyColorMap(relevance_scores, cv2.COLORMAP_JET)
heatmap_img = np.float32(heatmap_img) / 255
original_image = np.array(image.resize((224, 224))) / 255
superimposed_image = heatmap_img + original_image
superimposed_image = superimposed_image / np.max(superimposed_image)

# Save the superimposed image
superimposed_image_path = 'lionhabitat_lrp_superimposed.png'
cv2.imwrite(superimposed_image_path, np.uint8(255 * superimposed_image))

# Display the superimposed image
superimposed_img = Image.open(superimposed_image_path)
superimposed_img.show()
