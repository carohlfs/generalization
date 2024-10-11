import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

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

# Load the pretrained MobileNet v3 large model
model = mobilenet_v3_large(pretrained=True)
model.eval()

# Grad-CAM implementation
def grad_cam(model, input_tensor, index=None):
    gradients = None
    activations = None
    def save_gradients(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]
    def save_activations(module, input, output):
        nonlocal activations
        activations = output
    target_layer = model.features[-1][0]  # Last convolutional layer in MobileNet v3 large
    target_layer.register_forward_hook(save_activations)
    target_layer.register_backward_hook(save_gradients)
    output = model(input_tensor)
    if index is None:
        index = output.argmax(dim=1).item()
    one_hot = torch.zeros_like(output)
    one_hot[0, index] = 1
    model.zero_grad()
    output.backward(gradient=one_hot)
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    grad_cam = torch.sum(weights * activations, dim=1).squeeze()
    grad_cam = torch.relu(grad_cam)
    grad_cam = grad_cam.detach().cpu().numpy()
    # Normalize the Grad-CAM
    grad_cam = cv2.resize(grad_cam, (224, 224))
    grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())
    return grad_cam

# Compute Grad-CAM for the given image
grad_cam_heatmap = grad_cam(model, input_batch)

# Enhance the contrast of the heatmap
grad_cam_heatmap = np.uint8(255 * grad_cam_heatmap)
grad_cam_heatmap = cv2.equalizeHist(grad_cam_heatmap)

# Create a heatmap and save the image
plt.imshow(grad_cam_heatmap, cmap='hot', interpolation='nearest')
plt.axis('off')
plt.colorbar()
heatmap_path = 'lionhabitat_grad_cam_heatmap.png'
plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
plt.close()

# Display the saved heatmap
saved_heatmap = Image.open(heatmap_path)
saved_heatmap.show()

# Superimpose the heatmap on the original image
heatmap_img = cv2.applyColorMap(grad_cam_heatmap, cv2.COLORMAP_JET)
heatmap_img = np.float32(heatmap_img) / 255
original_image = np.array(image.resize((224, 224))) / 255
superimposed_image = heatmap_img + original_image
superimposed_image = superimposed_image / np.max(superimposed_image)

# Save the superimposed image
superimposed_image_path = 'lionhabitat_grad_cam_superimposed.png'
cv2.imwrite(superimposed_image_path, np.uint8(255 * superimposed_image))

# Display the superimposed image
superimposed_img = Image.open(superimposed_image_path)
superimposed_img.show()
