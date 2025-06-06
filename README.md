# vae
Implementation of a variational autoencoder

apt update && apt install vim -y && apt install screen -y

poetry init
poetry env use $(which python)
poetry add torch torchvision matplotlib numpy
poetry env info --path
Set interpreter in IDE (VS Code) equal to the `../bin/python3.10` path

# When restarting runpod
apt update && apt install vim -y && apt install screen -y && /workspace/miniconda3/bin/conda init && source ~/.bashrc && conda activate vae && wandb login
