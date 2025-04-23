import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now your regular imports
from ml_models.cyclegan import Generator
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Generator().to(device)

# Load weights (update path as needed)
weights_path = 'ai_models/G_A2B_50epochs.pth'
state_dict = torch.load(weights_path, map_location=device)

# Handle DataParallel if used during training
if all(k.startswith('module.') for k in state_dict.keys()):
    state_dict = {k[7:]: v for k,v in state_dict.items()}

model.load_state_dict(state_dict, strict=False)
model.eval()

# Prepare transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Test with a sample image (replace with your image path)
input_path = '/Users/prarabdhapandey/smoke-detector-app/uploads/enhanced_2023-01-29-00_00_2023-01-29-23_59_Sentinel-2_L2A_True_color.png'  # Put your test image in project root
output_path = 'test_output.png'

# Process image
with torch.no_grad():
    # Load and transform image
    image = Image.open(input_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Generate output
    output = model(image)
    
    # Convert back to PIL Image
    output = output.squeeze(0).cpu()
    output = (output * 0.5 + 0.5).clamp(0, 1)
    output_img = transforms.ToPILImage()(output)
    
    # Save and show result
    output_img.save(output_path)
    plt.imshow(output_img)
    plt.axis('off')
    plt.show()

print(f"Generated image saved to {output_path}")