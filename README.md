# vae
Implementation of a variational autoencoder

poetry init
poetry env use $(which python)
poetry add torch torchvision matplotlib numpy
poetry env info --path
Set interpreter in IDE (VS Code) equal to the `../bin/python3.10` path


echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc