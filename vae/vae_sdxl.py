import torch
import sys
import wandb
import os
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
from torchinfo import summary

from typing import Callable
from omegaconf import OmegaConf
from PIL import Image

torch.backends.cudnn.benchmark = True


import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock2D(nn.Module):
    """
    A basic ResNet block with two conv layers, GroupNorm and SiLU activation,
    plus a residual skip connection (with optional 1x1 projection).
    """
    def __init__(self, in_channels, out_channels, norm_groups=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_groups = norm_groups

        # First conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels)
        # Second conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels)
        # Projection if channels change
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = F.silu(self.norm1(out))
        out = self.conv2(out)
        out = self.norm2(out)
        if self.residual_conv is not None:
            identity = self.residual_conv(identity)
        out = out + identity
        out = F.silu(out)
        return out

class Downsample2D(nn.Module):
    """
    Downsamples spatial dims by 2 using strided convolution.
    """
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)

class Upsample2D(nn.Module):
    """
    Upsamples spatial dims by 2 using nearest neighbor + conv.
    """
    def __init__(self, channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class AutoencoderKL(nn.Module):
    """
    A convolutional VAE with spatial latents (e.g., (B, latent_channels, H/8, W/8)).
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        block_out_channels=(128, 256, 512, 512),
        latent_channels=4,
        norm_groups=32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        self.latent_channels = latent_channels
        self.norm_groups = norm_groups

        # Encoder: downsampling blocks
        self.encoder_blocks = nn.ModuleList()
        ch = in_channels
        for out_ch in block_out_channels:
            self.encoder_blocks.append(
                nn.Sequential(
                    ResnetBlock2D(ch, out_ch, norm_groups),
                    Downsample2D(out_ch)
                )
            )
            ch = out_ch

        # Final ResNet block before latent
        self.post_enc = ResnetBlock2D(ch, ch, norm_groups)

        # Latent projection convs for mu and logvar
        self.conv_mu = nn.Conv2d(ch, latent_channels, kernel_size=1)
        self.conv_logvar = nn.Conv2d(ch, latent_channels, kernel_size=1)

        # Decoder initial conv to expand latent
        self.conv_decode = nn.Conv2d(latent_channels, ch, kernel_size=1)

        # Decoder: upsampling blocks (reverse order)
        self.decoder_blocks = nn.ModuleList()
        for in_ch, out_ch in reversed(list(zip(block_out_channels, block_out_channels[1:] + (block_out_channels[-1],)))):
            self.decoder_blocks.append(
                nn.Sequential(
                    ResnetBlock2D(ch, in_ch, norm_groups),
                    Upsample2D(in_ch)
                )
            )
            ch = in_ch

        # Final ResNet block after upsampling
        self.post_dec = ResnetBlock2D(ch, ch, norm_groups)

        # Output conv to RGB
        self.conv_out = nn.Conv2d(ch, out_channels, kernel_size=1)

    def encode(self, x):
        """Encode input to spatial latent distribution (mu, logvar)."""
        h = x
        for block in self.encoder_blocks:
            h = block(h)
        h = self.post_enc(h)
        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        return mu, logvar

    def decode(self, z):
        """Decode latent z (sampled) to image space."""
        h = self.conv_decode(z)
        for block in self.decoder_blocks:
            h = block(h)
        h = self.post_dec(h)
        return self.conv_out(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = torch.sigmoid(self.decode(z))
        return recon, mu, logvar

# Example usage:
# vae = AutoencoderKL(in_channels=3, out_channels=3, sample_size=512)
# x = torch.randn(1, 3, 512, 512)
# recon, mu, logvar = vae(x)
# print(recon.shape, mu.shape, logvar.shape)



def vae_loss(input, target, mu : torch.Tensor, logvar : torch.Tensor, loss : Callable):
    # Analytic KL: -0.5 * (1 - log(sig^2) - mu^2 - e^(log(sig^2)))
    recon_loss = loss(input, target, reduction="sum") # F.binary_cross_entropy_with_logits()
    kl_div_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + kl_div_loss) / input.size(0)


def show_reconstructions(config, model, data_loader, num_images=8):
    model.eval()
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(config.device)
            if config.model_type == 'mlp': inputs = torch.flatten(inputs, start_dim=1)
            x_hat, _, _ = model(inputs)
            # x_hat = torch.sigmoid(x_hat)  # output logits → probability
            break  # take only one batch

    # reshape to (B, 1, 28, 28)
    inputs = inputs.cpu()
    inputs = inputs.view(
        -1, config.num_channels, config.image_size, config.image_size).cpu()
    x_hat = x_hat.view(
        -1, config.num_channels, config.image_size, config.image_size).cpu()

    # concat input/output for side-by-side view
    print(f"inputs.shape {inputs.shape}")
    print(f"x_hat.shape {x_hat.shape}")
    comparison = torch.cat([inputs[:num_images], x_hat[:num_images]])
    grid = vutils.make_grid(comparison, nrow=num_images)
    plt.figure(figsize=(15, 4))
    plt.axis("off")
    plt.title("Top: Original — Bottom: Reconstruction")
    plt.imshow(grid.permute(1, 2, 0).squeeze())
    plt.show()


def show_samples(config, model, latent_dim=16, num_images=8):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim).to(config.device)
        x_sample = model.decode(z)
        # x_sample = torch.sigmoid(x_sample).view(
        #     -1, config.num_channels, config.image_size, config.image_size).cpu()
        x_sample = x_sample.view(
            -1, config.num_channels,
            config.image_size, config.image_size
        ).cpu()

    grid = vutils.make_grid(x_sample, nrow=num_images)
    plt.figure(figsize=(15, 4))
    plt.axis("off")
    plt.title("Samples from Latent Space")
    plt.imshow(grid.permute(1, 2, 0).squeeze())
    plt.show()


def interpolate_latents(config, model, dataset, num_steps=10):
    model.eval()
    with torch.no_grad():
        # Select two random digits
        indices = np.random.choice(len(dataset), 2, replace=False)
        img1, _ = dataset[indices[0]]
        img2, _ = dataset[indices[1]]

        if config.model_type == "mlp":
            x1 = img1.view(1, -1).to(config.device)
            x2 = img2.view(1, -1).to(config.device)
        else:
            x1 = img1.unsqueeze(0).to(config.device) # (1, 3, 224, 224)
            x2 = img2.unsqueeze(0).to(config.device) # (1, 3, 224, 224)

        # Encode both to get means
        mu1, _ = model.encode(x1)
        mu2, _ = model.encode(x2)

        # Linearly interpolate in latent space
        interpolated = []
        for alpha in torch.linspace(0, 1, steps=num_steps):
            z = (1 - alpha) * mu1 + alpha * mu2
            x_hat = model.decode(z).view(
                1, config.num_channels,
                config.image_size, config.image_size
            )
            interpolated.append(x_hat.cpu())

        # Stack and visualize
        interpolated = torch.cat(interpolated, dim=0)
        grid = torchvision.utils.make_grid(interpolated, nrow=num_steps)
        plt.figure(figsize=(num_steps, 2))
        plt.axis("off")
        plt.title("Latent Interpolation")
        plt.imshow(grid.permute(1, 2, 0).squeeze())
        plt.show()


def get_train_val_dl(dataset, config):
    train_len = int(len(dataset) * config.p_train_len)

    train_ds, val_ds = random_split(dataset, [train_len, len(dataset) - train_len])

    train_dl = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        ### ADDED ####
        pin_memory=True,
        prefetch_factor=2,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    return train_dl, val_dl


def load_and_test_model(config):
    api = wandb.Api()
    artifact_name = config.artifact_name
    try:
        artifact = api.artifact(artifact_name, type='model')
        artifact_dir = artifact.download()
        raw_sd = torch.load(f"{artifact_dir}/best_model.pth", map_location="cpu")

        # strip out the "_orig_mod." prefix from every key
        fixed_sd = {}
        for k, v in raw_sd.items():
            # Python 3.9+: you can use k.removeprefix()
            new_k = k.removeprefix("_orig_mod.")
            fixed_sd[new_k] = v

        # Load model
        model = AutoencoderKL(
            in_channels=config.num_channels,
            out_channels=config.num_channels,
            latent_dims=config.latent_dims,
            p_dropout=config.p_dropout,
            image_size=config.image_size
        )
        # model.load_state_dict(torch.load(f"{artifact_dir}/best_model.pth", map_location="cpu"))
        model.load_state_dict(fixed_sd)
        model.eval()
        print("Model loaded successfully.")

        transform = T.Compose([
            T.CenterCrop(178),
            T.Resize(config.image_size),
            T.ToTensor(),
        ])
        dataset = CelebADataset(
            root_dir=os.path.join(config.data_root, "img_align_celeba"),
            transform=transform
        )
        _, val_dl = get_train_val_dl(dataset, config)
        print("Dataloaders created")

        show_reconstructions(config, model, val_dl)
        show_samples(config, model, latent_dim=config.latent_dims)
        interpolate_latents(config, model, dataset, num_steps=10)
    except wandb.CommError as e:
        print(f"Artifact not found: {artifact_name}")
        print(f"Error: {e}")
        exit(0)


class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_names = sorted([f for f in os.listdir(root_dir) if f.endswith(".jpg")])
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_names[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0


def train_test_model(config):
    config_dict = OmegaConf.to_container(config)
    wandb.init(
        project=config.project,
        name=config.name,
        config=config_dict,
        mode=config.wandb_mode,
    )
    transform = T.Compose([
        T.CenterCrop(178),
        T.Resize(config.image_size),
        T.ToTensor(),
    ])
    dataset = CelebADataset(
        root_dir=os.path.join(config.data_root, "img_align_celeba"),
        transform=transform
    )

    train_dl, val_dl = get_train_val_dl(dataset, config)
    model = AutoencoderKL(
        in_channels=config.num_channels,
        out_channels=config.num_channels,
    )
    model = torch.compile(model)
    model = model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    print("Training Start")

    ### ADDED ###
    best_val_loss = float('inf')
    train_epoch_loss = 0.0
    val_epoch_loss = 0.0
    for epoch in range(1, config.n_epochs + 1):
        tqdm.write(f"Epoch {epoch}/{config.n_epochs + 1}")
        model.train()
        with tqdm(train_dl, desc="Training") as pbar:
            train_loss = 0.0
            for batch_idx, (inputs, _) in enumerate(pbar):
                inputs = inputs.to(config.device)
                # inputs = torch.flatten(inputs, start_dim=1)
                recons, mu, logvar = model(inputs)
                optimizer.zero_grad()
                # loss = vae_loss(recons, inputs, mu, logvar, F.binary_cross_entropy_with_logits)

                #### ADDED #####
                loss = vae_loss(recons, inputs, mu, logvar, F.binary_cross_entropy)
                loss.backward()

                ###### ADDED ######
                #### Gradient Clipping #####
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix(train_loss=f"{train_loss / (batch_idx + 1):.2f}")
        train_epoch_loss = train_loss / len(train_dl)
        tqdm.write(f"Train Loss {train_epoch_loss:.2f}")

        with tqdm(val_dl, desc="Validation") as pbar:
            ###### ADDED ######
            with torch.no_grad():
                model.eval()
                val_loss = 0.0
                for batch_idx, (inputs, _) in enumerate(pbar):
                    inputs = inputs.to(config.device)
                    # inputs = torch.flatten(inputs, start_dim=1)
                    recons, mu, logvar = model(inputs)
                    # loss = vae_loss(recons, inputs, mu, logvar, F.binary_cross_entropy_with_logits)
                    loss = vae_loss(recons, inputs, mu, logvar, F.binary_cross_entropy)
                    val_loss += loss.item()
                    pbar.set_postfix(val_loss=f"{val_loss / (batch_idx + 1) :.2f}")
            val_epoch_loss = val_loss / len(val_dl)
        tqdm.write(f"val Loss {val_epoch_loss:.2f}")

        ###### ADDED ######
        if config.save_model and (val_epoch_loss < best_val_loss):
            best_val_loss = val_epoch_loss
            tqdm.write("Writing best model...")
            torch.save(model.state_dict(), "best-model.pth")
            artifact = wandb.Artifact(name=f"{config.name}-best-model", type="model")
            artifact.add_file("best-model.pth")
            wandb.log_artifact(artifact)
            tqdm.write("Model written.")

        ###### ADDED ######
        wandb.log({
            "epoch": epoch,
            "train/loss": train_epoch_loss,
            "val/loss": val_epoch_loss,
        })
    
    show_reconstructions(config, model, val_dl, num_images=8)
    show_samples(config, model, latent_dim=16, num_images=8)
    interpolate_latents(config, model, dataset, num_steps=10)


def main():
    env = os.environ.get("ENV", "local")
    print(f"env={env}")
    config = load_config(env)
    print("Configuration loaded")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.device, config.env = device, env
    print(f"Seed {config.seed} Device {config.device}")

    if config.summary:
        m = AutoencoderKL(in_channels=config.num_channels, out_channels=config.num_channels)
        input_shape = (1, config.num_channels, config.image_size, config.image_size)
        summary(
            m, 
            input_size=input_shape,
            col_names=["input_size", "output_size", "num_params"],
            # verbose=2
        )
        exit()

    if config.load_and_test_model:
        load_and_test_model(config)
    elif config.train_model:
        train_test_model(config)


def load_config(env="local"):
    base_config = OmegaConf.load("config/base.yaml")

    env_path = f"config/{env}.yaml"
    if os.path.exists(env_path):
        env_config = OmegaConf.load(env_path)
        # Merges env_config into base_config (env overrides base)
        config = OmegaConf.merge(base_config, env_config)
    else:
        config = base_config

    return config


if __name__ == '__main__':
    main()