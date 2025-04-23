import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import os

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Load model (use relative path from your project structure)
model_path = os.path.join('backend', 'ai_models', 'srgan_generator.pth')
generator = Generator()
generator.load_state_dict(torch.load(model_path, map_location='cpu'))
generator.eval()

def generate_enhanced_image(image_path):
    """Process image through SRGAN model"""
    pil_image = Image.open(image_path).convert('RGB')
    
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Downscale to simulate low-res input
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    # Process image
    lr_tensor = transform(pil_image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        sr_tensor = generator(lr_tensor)
    
    # Convert back to PIL Image
    sr_tensor = sr_tensor.squeeze(0).cpu()
    sr_tensor = sr_tensor * 0.5 + 0.5  # Denormalize
    sr_image = transforms.ToPILImage()(sr_tensor.clamp(0, 1))
    
    return sr_image