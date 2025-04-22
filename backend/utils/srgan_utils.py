import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

# Your Generator class (with correct __init__)
class Generator(nn.Module):
    def __init__(self):  # ✅ Fixed
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

# Load model
generator = Generator()
generator.load_state_dict(torch.load('/Users/prarabdhapandey/smoke-detector-app/backend/models/srgan_generator.pth', map_location='cpu'))
generator.eval()

# Inference function
def apply_srgan(pil_image):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Simulate low-res input
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    lr = transform(pil_image).unsqueeze(0)  # Add batch dim

    with torch.no_grad():
        sr = generator(lr)

    sr = sr.squeeze(0).detach().cpu()
    sr = sr * 0.5 + 0.5  # Denormalize
    sr_image = transforms.ToPILImage()(sr.clamp(0, 1))

    return sr_image