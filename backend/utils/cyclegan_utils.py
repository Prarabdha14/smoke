import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Generator architecture (same as your training one)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Load trained generator
device = torch.device("cpu")  # or "cuda" if available
generator = Generator()
generator.load_state_dict(torch.load('/Users/prarabdhapandey/smoke-detector-app/backend/models/G_A2B_50epochs.pth', map_location=device))
generator.to(device)
generator.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# CycleGAN inference function
def apply_cyclegan(pil_image):
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        fake = generator(input_tensor)

    fake = fake.squeeze(0).cpu() * 0.5 + 0.5  # Denormalize
    output_image = transforms.ToPILImage()(fake.clamp(0, 1))

    return output_image