import torch
from models.network_swinir import SwinIR as net

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
        resi_connection='1conv'
    )
    param_key_g = 'params_ema'
    return model, param_key_g

model_path = "experiments/pretrained_models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
model, param_key_g = define_model(scale=4)

pretrained_model = torch.load(model_path)
model.load_state_dict(pretrained_model[param_key_g], strict=True)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(model)

batch_size = 4
channels = 3
height = 64
width = 64 # as printed by model
example_input = torch.randn(batch_size, channels, height, width, device=device)

# Step 3: Trace the model
traced_model = torch.jit.trace(model, example_input)

# Step 4: Save the traced model
output_path = "swinir_real_sr_traced.pt"
traced_model.save(output_path)

print(f"Traced model saved to {output_path}")



