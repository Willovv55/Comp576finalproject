import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from model import UNet
from diffusion import train, sample

# ========== Parameters ==========
T = 1000
EPOCHS = 15000
IMG_SIZE = 512
BATCH_SIZE = 1
INPUT_DIR = 'data/'
OUTPUT_DIR = 'outputs/'
device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== Image preprocessing ==========
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ========== Load the UNet model ==========
model = UNet(channels=3).to(device)

# ========== Create the output directory if it doesn't exist ==========
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== Process each image in the input folder ==========
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        img_path = os.path.join(INPUT_DIR, filename)

        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = img.clamp(0., 1.)

        print(f"Training on image: {filename}")
        train(model, img, epochs=EPOCHS, T=T, save_every=2000)

        print(f"Sampling on image: {filename}")
        out = sample(model, shape=(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE), T=T)

        # Save the denoised output image
        out_path = os.path.join(OUTPUT_DIR, f'denoised_{filename}')
        save_image(out, out_path)
        print(f"Saved denoised image: {out_path}")
