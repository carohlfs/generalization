import torch
import torchvision.models as models

# List of model names
model_names = [
    'resnext101_32x8d', 'wide_resnet101_2', 'efficientnet_b0', 'efficientnet_b7',
    'resnet101', 'densenet201', 'vgg19_bn', 'mobilenet_v3_large', 'mobilenet_v3_small',
    'googlenet', 'inception_v3', 'alexnet'
]

# Function to calculate the norm of the weights
def calculate_weight_norm(model):
    total_norm = 0
    for param in model.parameters():
        total_norm += param.norm().item() ** 2
    return total_norm ** 0.5

# Dictionary to store the norms
weight_norms = {}

# Load each model and calculate the norm of its weights
for name in model_names:
    model = getattr(models, name)(pretrained=True)
    model.eval()
    norm = calculate_weight_norm(model)
    weight_norms[name] = norm
    print(f'{name}: {norm}')

# If you want to print all the norms at once
print("\nWeight norms for pretrained models:")
for name, norm in weight_norms.items():
    print(f'{name}: {norm}')
