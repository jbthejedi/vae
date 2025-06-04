import torch
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

from omegaconf import OmegaConf

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VAE(nn.Module):
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


def vae_loss(x, x_hat, mu, logvar):
    recon_loss = F.binary_cross_entropy_with_logits(x_hat, x, reduction="sum")
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


def show_reconstructions(config, model, data_loader, num_images=8):
    model.eval()
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
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
        z = torch.randn(num_images, latent_dim).to(device)
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

        x1 = img1.view(1, -1).to(device)
        x2 = img2.view(1, -1).to(device)

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
    transform = T.Compose([
        T.Resize((config.image_size, config.image_size)),
        T.ToTensor(),
    ])
    dataset = datasets.CIFAR10(
        root=config.data_root,
        download=False,
        transform=transform,
    )
    # transform = T.Compose([
    #     T.ToTensor(),
    # ])
    # dataset = datasets.MNIST(
    #     root='/Users/justinbarry/projects/vision-transformer/data',
    #     download=False,
    #     transform=transform,
    # )
    print("Dataset loaded")

    train_dl = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    print("Loading model")
    input_dims = config.num_channels * config.image_size * config.image_size
    model = VAE(input_dims)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    print("Training Start")
    best_train_loss = 'inf'
    for epoch in range(1, config.n_epochs+1):
        tqdm.write(f"Epoch {epoch}/{config.n_epochs+1}")
        model.train()
        with tqdm(train_dl, desc="Training") as pbar:
            for inputs, _ in pbar:
                inputs = inputs.to(device)
                inputs = torch.flatten(inputs, start_dim=1)
                optimizer.zero_grad()
                x_hat, mu, logvar = model(inputs)
                loss = vae_loss(inputs, x_hat, mu, logvar)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * inputs.shape(0)
                train_total += inputs.shape(0)
                pbar.set_postfix(batch_loss=loss.item())
        train_epoch_loss = total_loss / train_total
        if train_epoch_loss < best_train_loss: best_train_loss = train_epoch_loss
        if config.device == 'cuda' and config.save_model and train_epoch_loss > best_train_loss:
            tqdm.write("Writing best model...")
            best_val_dice = train_epoch_loss
            torch.save(model.state_dict(), "best_model.pth")
            artifact = wandb.Artifact(name=f"{config.name}_best_model", type="model")
            artifact.add_file("best_model.pth")
            wandb.log_artifact(artifact)
            tqdm.write("Model written.")

        wandb.log({
            "epoch": epoch,
            "train/loss": train_epoch_loss,
        })
    
    if config.local_visualization:
        show_reconstructions(config, model, train_dl)
        # show_samples(config, model, latent_dim=16, num_images=10)
        interpolate_latents(config, model, dataset, num_steps=10)


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


def main():
    env = os.environ.get("ENV", "local")
    config = load_config(env)

    # mu1, mu2 = 1, 3
    # num_steps = 10
    # for alpha in torch.linspace(0, 1, steps=num_steps):
    #     z = (1 - alpha) * mu1 + alpha * mu2

    train_test_model(config)


if __name__ == '__main__':
    main()