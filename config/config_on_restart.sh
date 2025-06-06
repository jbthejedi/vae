#!/bin/bash

source ~/.bashrc

set -e  # Exit on error

echo "==> Updating system package list"
apt update

echo "==> Installing vim if not already installed"
if ! command -v vim &> /dev/null; then
    apt install -y vim
else
    echo "vim already installed"
fi

echo "==> Installing screen if not already installed"
if ! command -v screen &> /dev/null; then
    apt install -y screen
else
    echo "screen already installed"
fi

echo "==> Installing wget if not already installed"
if ! command -v wget &> /dev/null; then
    apt install -y wget
else
    echo "wget already installed"
fi

echo "==> Authenticating GitHub CLI (manual login required if not done already)"
gh auth status || gh auth login

echo "==> Cloning your repo"
read -p "Enter GitHub repo name (e.g. "vae"): " REPO
REPO_NAME=$(basename "$REPO")
if [ -d "$REPO_NAME" ]; then
    echo "Repo '$REPO_NAME' already exists. Skipping clone."
    cd "$REPO_NAME"
else
    gh repo clone "$REPO"
    cd "$REPO_NAME"
fi

echo 'export PATH="/workspace/poetry/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

poetry env use python3.10
poetry install

echo "==> Logging in to Weights & Biases (manual login required if not done)"
wandb status || poetry run wandb login

echo "✅ Restart configuration setup complete."
