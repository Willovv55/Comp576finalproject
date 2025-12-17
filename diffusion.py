import torch
import torch.nn.functional as F
import numpy as np
import os
from torchvision.utils import save_image

def get_noise_schedule(T):
    betas = np.linspace(1e-5, 0.01, T)
    alphas = 1 - betas
    alphas_cumprod = np.cumprod(alphas)
    return torch.tensor(betas, dtype=torch.float32), torch.tensor(alphas_cumprod, dtype=torch.float32)

def add_noise(x0, t, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod):
    noise = torch.randn_like(x0)
    sqrt_alpha = sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
    xt = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
    return xt, noise

def train(model, data, epochs=15000, T=1000, save_every=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    betas, alphas_cumprod = get_noise_schedule(T)
    sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod).to(data.device)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod).to(data.device)

    os.makedirs('outputs', exist_ok=True)

    for epoch in range(epochs):
        model.train()
        t = torch.randint(0, T, (data.shape[0],)).long().to(data.device)
        xt, noise = add_noise(data, t, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod)
        pred_noise = model(xt, t.float() / T)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

        if loss.item() > 1.1:
            print("Loss exploded, stop training!")
            break

        if epoch % save_every == 0 and epoch > 0:
            denoised = sample(model, shape=data.shape, T=T)
            save_image(denoised, f'outputs/denoised_epoch{epoch}.png')

@torch.no_grad()
def sample(model, shape, T=1000):
    betas, alphas_cumprod = get_noise_schedule(T)
    sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod)

    xt = torch.randn(shape).to(next(model.parameters()).device)
    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, dtype=torch.float32).to(xt.device)
        pred_noise = model(xt, t_batch / T)

        sqrt_alpha = sqrt_alpha_cumprod[t]
        xt = (xt - (1 - sqrt_alpha**2).sqrt() * pred_noise) / sqrt_alpha

        if t > 0:
            noise = torch.randn_like(xt)
            beta = betas[t]
            xt += beta.sqrt() * noise

        xt = xt.clamp(0., 1.) 

    return xt
