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
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchinfo import summary

from omegaconf import OmegaConf

class VAELinear(nn.Module):
    def __init__(
        self,
        input_dim=784,
        hidden_dim=256,
        latent_dim=16,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def decode(self, x):
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, p_dropout):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout)
        )

    def forward(self, x):
        return self.double_conv(x)


class VAEUnet(nn.Module):
    def __init__(self, in_channels, latent_dims, p_dropout, image_size):
        super().__init__()
        """
        Assumptions for dimension calculations
        H_in, W_in = 224
        Conv:          {h/w}_out = [({h/w}_in + 2p - d(k - 1) - 1) / s] + 1
        ConvTranspose: {h/w}_out = ({h/w}_in - 1)s - 2p + k + output_padding
        """
        self.image_size = image_size
        self.layer0 = nn.Sequential( # -> (64, 75, 75)
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=3, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )
        self.pool0 = nn.MaxPool2d(2) # -> (64, 37, 37)

        self.conv1 = DoubleConv(64, 128, p_dropout) # -> (128, 37, 37)
        # MaxPool2d stride=2 by default
        self.pool1 = nn.MaxPool2d(2) # -> (128, 18, 18)

        self.conv2 = DoubleConv(128, 256, p_dropout) # -> (256, 18, 18)
        self.pool2 = nn.MaxPool2d(2) # -> (256, 9, 9)

        self.conv3 = DoubleConv(256, 512, p_dropout) # -> (512, 9, 9)

        C, H, W = self._get_flattened_shape(in_channels)
        self.flattened_dim = C * H * W
        self.fc_mu = nn.Linear(self.flattened_dim, latent_dims) # -> (512*9*9, 64)
        self.fc_logvar = nn.Linear(self.flattened_dim, latent_dims) # -> (512*9*9, 64)

        self.fc_up = nn.Linear(latent_dims, C*W*H) # -> (64, 512*9*9)

        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2) # -> (256, 18, 18)
        self.refine2 = DoubleConv(512, 256, p_dropout)

        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2, output_padding=1) # -> (128, 36, 36)
        self.refine1 = DoubleConv(256, 128, p_dropout)

        self.up0 = nn.ConvTranspose2d(128, 64, 2, stride=2, output_padding=1) # -> (, 64, 75, 75)
        self.refine0 = DoubleConv(128, 64, p_dropout)

        self.head = nn.ConvTranspose2d(64, 3, 8, 3, 3)


    def _get_flattened_shape(self, in_channels):
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, self.image_size, self.image_size)
            d0 = self.layer0(dummy)
            x = self.pool0(d0)

            d1 = self.conv1(x)
            x = self.pool1(d1)

            d2 = self.conv2(x)
            x = self.pool2(d2)

            m = self.conv3(x)
            _, C, H, W = m.shape

            return C, H, W

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        d0 = self.layer0(x)
        x = self.pool0(d0)

        d1 = self.conv1(x)
        x = self.pool1(d1)

        d2 = self.conv2(x)
        x = self.pool2(d2)

        m = self.conv3(x)

        B, C, W, H = m.shape
        # print(f"B, C, W, H {B, C, W, H}")
        m = m.view(-1, C*W*H)
        mu = self.fc_mu(m)
        logvar = self.fc_logvar(m)

        z = self.reparameterize(mu, logvar) # -> (,64)
        z = self.fc_up(z) # (, 512*9*9)

        z = z.view(-1, C, W, H)

        up2 = self.refine2(torch.cat([self.up2(z), d2], dim=1))
        up1 = self.refine1(torch.cat([self.up1(up2), d1], dim=1))
        up0 = self.refine0(torch.cat([self.up0(up1), d0], dim=1))
        out = self.head(up0) 

        return out, mu, logvar


def vae_loss(x, x_hat, mu, logvar):
    recon_loss = F.binary_cross_entropy_with_logits(x_hat, x, reduction="sum")
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


def show_reconstructions(config, model, data_loader, num_images=8):
    model.eval()
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(config.device)
            inputs_flat = torch.flatten(inputs, start_dim=1)
            x_hat, _, _ = model(inputs_flat)
            x_hat = torch.sigmoid(x_hat)  # output logits → probability
            break  # take only one batch

    # reshape to (B, 1, 28, 28)
    inputs = inputs.cpu()
    x_hat = x_hat.view(
        -1, config.num_channels, config.image_size, config.image_size).cpu()

    # concat input/output for side-by-side view
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
        x_sample = torch.sigmoid(x_sample).view(
            -1, config.num_channels, config.image_size, config.image_size).cpu()

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

        x1 = img1.view(1, -1).to(config.device)
        x2 = img2.view(1, -1).to(config.device)

        # Encode both to get means
        mu1, _ = model.encode(x1)
        mu2, _ = model.encode(x2)

        # Linearly interpolate in latent space
        interpolated = []
        for alpha in torch.linspace(0, 1, steps=num_steps):
            z = (1 - alpha) * mu1 + alpha * mu2
            x_hat = torch.sigmoid(model.decode(z)).view(
                1, config.num_channels, config.image_size, config.image_size)
            interpolated.append(x_hat.cpu())

        # Stack and visualize
        interpolated = torch.cat(interpolated, dim=0)
        grid = torchvision.utils.make_grid(interpolated, nrow=num_steps)
        plt.figure(figsize=(num_steps, 2))
        plt.axis("off")
        plt.title("Latent Interpolation")
        plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
        plt.show()


def train_test_model(config):
    config_dict = OmegaConf.to_container(config)
    wandb.init(
        project=config.project,
        name=config.name,
        config=config_dict,
        mode=config.wandb_mode,
    )
    if config.dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(
            root=config.data_root,
            download=False,
            transform=T.Compose([
                T.Resize((config.image_size, config.image_size)),
                T.ToTensor(),
            ])
        )
    
    if config.dataset_name == 'minst':
        dataset = datasets.MNIST(
            root='/Users/justinbarry/projects/vision-transformer/data',
            download=False,
            transform=T.Compose([
                T.ToTensor(),
            ]),
        )
    print("Dataset loaded")

    train_len = int(len(dataset) * config.p_train_len)
    train_ds, val_ds = random_split(dataset, [train_len, len(dataset) - train_len])

    train_dl = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    print("Loading model")
    if config.model_type == 'mlp':
        input_dims = config.num_channels * config.image_size * config.image_size
        model = VAELinear(input_dims)
    if config.model_type == 'unet':
        model = VAEUnet(
            in_channels=3,
            latent_dims=64,
            p_dropout=config.p_dropout,
            image_size=config.image_size
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    print("Training Start")
    best_val_loss = float('inf')
    train_epoch_loss = 0.0
    val_epoch_loss = 0.0
    for epoch in range(1, config.n_epochs+1):
        tqdm.write(f"Epoch {epoch}/{config.n_epochs+1}")
        model.train()
        total_loss = 0.0
        train_total = 0
        with tqdm(train_dl, desc="Training") as pbar:
            for inputs, _ in pbar:
                inputs = inputs.to(config.device)
                if config.model_type == 'mlp':
                    inputs = torch.flatten(inputs, start_dim=1)
                optimizer.zero_grad()
                x_hat, mu, logvar = model(inputs)
                loss = vae_loss(inputs, x_hat, mu, logvar)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                train_total += inputs.size(0)

                pbar.set_postfix(batch_loss=loss.item())
        train_epoch_loss = total_loss / train_total
        tqdm.write(f"Train Loss: {train_epoch_loss}")

        with tqdm(val_dl, desc="Validation") as pbar:
            model.eval()
            with torch.no_grad():
                total_loss = 0.0
                val_total = 0.0
                for inputs, _ in pbar:
                    inputs = inputs.to(config.device)
                    if config.model_type == 'mlp':
                        inputs = torch.flatten(inputs, start_dim=1)
                    x_hat, mu, logvar = model(inputs)
                    loss = vae_loss(inputs, x_hat, mu, logvar)
                    total_loss += loss.item() * inputs.size(0)
                    val_total += inputs.size(0)
                    pbar.set_postfix(batch_loss=loss.item())
                val_epoch_loss = total_loss / val_total
        tqdm.write(f"Val Loss: {val_epoch_loss}")

        if config.save_model and (val_epoch_loss < best_val_loss):
            best_val_loss = val_epoch_loss
            tqdm.write("Writing best model...")
            torch.save(model.state_dict(), "best_model.pth")
            artifact = wandb.Artifact(name=f"{config.name}_best_model", type="model")
            artifact.add_file("best_model.pth")
            wandb.log_artifact(artifact)
            tqdm.write("Model written.")

        wandb.log({
            "epoch": epoch,
            "train/loss": train_epoch_loss,
            "val/loss": val_epoch_loss,
        })
    
    if config.local_visualization:
        show_reconstructions(config, model, train_dl)
        # show_samples(config, model, latent_dim=16, num_images=10)
        interpolate_latents(config, model, dataset, num_steps=10)


def load_config(env):
    base_config = OmegaConf.load("config/base.yaml")

    env_path = f"config/{env}.yaml"
    if os.path.exists(env_path):
        env_config = OmegaConf.load(env_path)
        # Merges env_config into base_config (env overrides base)
        config = OmegaConf.merge(base_config, env_config)
    else:
        config = base_config

    return config

def load_and_test_model(config):
    api = wandb.Api()
    artifact_name = config.artifact_name
    try:
        artifact = api.artifact(artifact_name, type='model')
        artifact_dir = artifact.download()

        # Load model
        input_dims = config.num_channels * config.image_size * config.image_size
        model = VAELinear(input_dims).to(config.device)
        model.load_state_dict(torch.load(f"{artifact_dir}/best_model.pth", map_location="cpu"))
        model.eval()
        print("Model loaded successfully.")

        dataset = datasets.CIFAR10(
            root=config.data_root,
            download=False,
            transform=T.Compose([
                T.Resize((config.image_size, config.image_size)),
                T.ToTensor(),
            ])
        )
        print("Dataset loaded")
        train_dl = DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        show_reconstructions(config, model, train_dl)
        interpolate_latents(config, model, dataset, num_steps=10)
    except wandb.CommError as e:
        print(f"Artifact not found: {artifact_name}")
        print(f"Error: {e}")
        exit(0)

def get_env_from_argv(default="local"):
    for arg in sys.argv:
        if arg.startswith("env="):
            return arg.split("=")[1]
    print(f"default {default}")
    return default

def main():
    env = get_env_from_argv()
    config = load_config(env)
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # m = VAEUnet(3, 64, p_dropout=config.p_dropout)
    # summary(
    #     m, 
    #     input_size=(1, 3, 224, 224),  # Add batch size here!
    #     col_names=["input_size", "output_size", "num_params"],
    #     verbose=2
    # )

    # exit()

    # mu1, mu2 = 1, 3
    # num_steps = 10
    # for alpha in torch.linspace(0, 1, steps=num_steps):
    #     z = (1 - alpha) * mu1 + alpha * mu2

    if config.load_and_test_model:
        load_and_test_model(config)
    elif config.train_model:
        train_test_model(config)


if __name__ == '__main__':
    main()