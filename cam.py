import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
import json

# Load the pre-trained MobileNetV3 Large model
model = mobilenet_v3_large(pretrained=True)
model.eval()

# Load ImageNet class index
with open("imagenet_class_index.json") as f:
    class_idx = json.load(f)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

# Define the image preprocessing function
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the image
img_path = 'lioncity.png'
img = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format
img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension

# Make a prediction to ensure the model is working correctly
with torch.no_grad():
    predictions = model(img_tensor)
    predicted_probs = torch.nn.functional.softmax(predictions[0], dim=0)
    top_probs, top_idxs = torch.topk(predicted_probs, 3)
    top_classes = [top_idxs[i].item() for i in range(3)]
    top_probs = [top_probs[i].item() * 100 for i in range(3)]  # Convert to percentages
    top_class_names = [idx2label[idx] for idx in top_classes]
    print('Top predicted classes:', top_class_names)
    print('Top predicted probabilities:', top_probs)

# Define a function to generate CAM
def generate_cam(model, img_tensor, class_idx):
    def hook_fn(module, input, output):
        global features
        features = output
        features.retain_grad()
    handle = model.features[-1][-1].register_forward_hook(hook_fn)  # Adjust hook for last conv layer
    img_tensor.requires_grad_()
    output = model(img_tensor)
    class_score = output[0, class_idx]
    handle.remove()
    model.zero_grad()
    class_score.backward(retain_graph=True)
    gradients = features.grad  # Get gradients for the features
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(features.size(1)):
        features[0, i, :, :] *= pooled_gradients[i]
    cam = torch.mean(features, dim=1).squeeze()
    cam = F.relu(cam)
    cam = cam - torch.min(cam)
    cam = cam / torch.max(cam)
    cam = cam.detach().numpy()
    return cam

# Generate CAMs for top predicted classes
cams = [generate_cam(model, img_tensor, idx) for idx in top_classes]

# Visualize the CAMs along with the original image and class probabilities
fig, ax = plt.subplots(1, 4, figsize=(20, 5))

# Original image
ax[0].imshow(img)
ax[0].axis('off')
ax[0].set_title('Original Image')

# Plot CAMs for top 3 classes
for i in range(3):
    ax[i+1].imshow(img)
    ax[i+1].imshow(cams[i], cmap='jet', alpha=0.5)
    ax[i+1].axis('off')
    ax[i+1].set_title(f'Class: {top_class_names[i]}, Prob: {top_probs[i]:.2f}%')

# Save the figure
output_path = 'lioncity_cam_output.png'
plt.savefig(output_path)
print(f'CAM output saved to {output_path}')
