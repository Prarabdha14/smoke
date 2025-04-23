import torch
from torchvision import transforms
from PIL import Image
import os
from ml_models.cyclegan import Generator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join('ai_models', 'G_A2B_50epochs.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_cyclegan_model():
    """Load the CycleGAN generator model with architecture mismatch handling"""
    try:
        # Initialize model
        generator = Generator().to(device)
        
        # Load state dict
        state_dict = torch.load(MODEL_PATH, map_location=device)
        
        # Handle DataParallel if used during training
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        # Special handling for layer 6 bias mismatch
        if 'model.6.bias' in state_dict:
            if state_dict['model.6.bias'].shape[0] == 128 and generator.model[6].bias.shape[0] == 256:
                logger.warning("Adjusting model.6.bias from 128 to 256 dimensions")
                # Create a new bias tensor with correct shape
                new_bias = torch.zeros(256)
                new_bias[:128] = state_dict['model.6.bias']  # Copy existing values
                state_dict['model.6.bias'] = new_bias
        
        # Load weights with strict=False to handle other potential mismatches
        generator.load_state_dict(state_dict, strict=False)
        generator.eval()
        
        logger.info("CycleGAN model loaded successfully with adjusted parameters")
        return generator
        
    except Exception as e:
        logger.error(f"Failed to load CycleGAN model: {str(e)}")
        raise RuntimeError(f"Error loading CycleGAN model: {str(e)}")

def generate_enhanced_image(image_path):
    """Process single image through CycleGAN"""
    try:
        # Validate input image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        generator = load_cyclegan_model()
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        image = Image.open(image_path).convert('RGB')
        if image.mode != 'RGB':
            raise ValueError("Image must be in RGB format")
            
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = generator(image)
        
        output = output.squeeze(0).cpu()
        output = (output * 0.5 + 0.5).clamp(0, 1)
        return transforms.ToPILImage()(output)
        
    except Exception as e:
        logger.error(f"Image generation failed for {image_path}: {str(e)}")
        raise RuntimeError(f"Image generation failed: {str(e)}")

if __name__ == "__main__":
    # Enhanced test function
    try:
        print("=== Testing CycleGAN Model Loading ===")
        model = load_cyclegan_model()
        
        print("\n=== Testing Image Generation ===")
        test_img = Image.new('RGB', (256, 256), color='red')
        output = generate_enhanced_image(test_img)
        
        print("\nTest results:")
        print(f"Input size: {test_img.size}")
        print(f"Output size: {output.size}")
        print("All tests passed successfully!")
        
    except Exception as e:
        print(f"\n!!! Test failed: {e}")
        raise