import torch
import torchvision.transforms as transforms
from models.network_swinir import SwinIR as net
from PIL import Image

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 2

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        # transforms.Resize((128, 128)),  # Resize to the model's expected input size
        # transforms.Resize((64, 64)), 
        transforms.ToTensor(),
    ])
    # xx = torch.cat([transform(img).unsqueeze(0).float()]*2, dim=0) 
    xx = torch.cat([transform(img).unsqueeze(0).float()]*batch_size, dim=0) 
    input(f'{xx.shape = }')
    return  xx # Add batch dimension

class WrapperModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.inference_mode()
    def forward(self, x: torch.Tensor):
        x = x.contiguous()
        y = self.model(x)   
        return y.contiguous()

def define_model(scale=4):
    model = net(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='nearest+conv',
        resi_connection='1conv',
    )
    return model



# Define and load the model
model_path = "experiments/pretrained_models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
model = define_model(scale=4)

model = model.to(device)
param_key_g = 'params_ema'
pretrained_model = torch.load(model_path, map_location=device)
model.load_state_dict(pretrained_model[param_key_g], strict=True)
model.eval()
model = WrapperModel(model)
model = model.eval()

# Load specific image and trace model on it
image_path = "/home/ubuntu/members/ayush/super-res/SwinIR/img.jpeg"  # Update this with your actual image path
input_tensor = load_image(image_path).to(device)
input(f'{input_tensor.shape = }')

# # Trace the model with this specific image
traced_model = torch.jit.trace(model, input_tensor)
output_path = "swinir_real_sr_traced.pt"
traced_model.save(output_path)
print(f"Traced model saved to {output_path}")

# exit()
import cv2
import numpy as np
output_path = "swinir_real_sr_traced.pt"
traced_model = torch.jit.load(output_path).to(device)
img = load_image("img.jpeg").to(device) # DO NOT RESIZE

input(f'{img.shape = }')
with torch.no_grad():
    output_tensor = traced_model(img)
from PIL import Image
import numpy as np
for i,outputs in enumerate(output_tensor):
    input(f'{outputs.shape = } {outputs.max()} {outputs.min()}')
    outputs = outputs.clamp_(0, 1)
    input(f'{outputs.shape = } {outputs.max()} {outputs.min()}')
    output_image = transforms.ToPILImage()(outputs.squeeze(0).cpu())
    # input(type(output_image))
    output_image_path = f"output_image_{i}.jpg"
    output_image.save(output_image_path)
    print(f"Output image saved to {output_image_path}")
