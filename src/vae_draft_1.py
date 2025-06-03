import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def show_reconstructions(model, data_loader, num_images=8):
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
    x_hat = x_hat.view(-1, 1, 28, 28).cpu()

    # concat input/output for side-by-side view
    comparison = torch.cat([inputs[:num_images], x_hat[:num_images]])
    grid = vutils.make_grid(comparison, nrow=num_images)
    plt.figure(figsize=(15, 4))
    plt.axis("off")
    plt.title("Top: Original — Bottom: Reconstruction")
    plt.imshow(grid.permute(1, 2, 0).squeeze())
    plt.show()


def show_samples(model, latent_dim=16, num_images=8):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim).to(device)
        x_sample = model.decode(z)
        x_sample = torch.sigmoid(x_sample).view(-1, 1, 28, 28).cpu()

    grid = vutils.make_grid(x_sample, nrow=num_images)
    plt.figure(figsize=(15, 4))
    plt.axis("off")
    plt.title("Samples from Latent Space")
    plt.imshow(grid.permute(1, 2, 0).squeeze())
    plt.show()


def main():
    image_size = 28
    transform = T.Compose([
        T.ToTensor(),
    ])
    dataset = datasets.CIFAR10(
        root='/Users/justinbarry/projects/vision-transformer/data',
        download=False,
        transform=transform,
    )
    # dataset = datasets.MNIST(
    #     root='/Users/justinbarry/projects/vision-transformer/data',
    #     download=False,
    #     transform=transform,
    # )
    print("Dataset downloaded")
    batch_size = 64
    train_dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    print("Loading model")
    model = VAE(input_dim=image_size*image_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("Training Start")
    for epoch in range(1, 11):
        tqdm.write(f"Epoch {epoch}/{11}")
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

                pbar.set_postfix(loss=loss.item())
    
    show_reconstructions(model, train_dl)
    show_samples(model, latent_dim=16, num_images=10)


if __name__ == '__main__':
    main()