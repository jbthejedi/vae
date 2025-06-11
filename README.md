# vae
Implementation of a variational autoencoder

# Run before running setup_env.sh to use vim
apt update && apt install vim -y && apt install screen -y

# SCP example
scp -P 42260 -i ~/.ssh/id_ed25519 cifar-10-python.tar.gz root@195.26.233.33:/workspace/data
tar --no-same-owner -xvf cifar-10-python.tar.gz

### Setup
Run config/config_on_restart.sh

# Ensure poetry venvs are created in Project directores
# venv create directory
poetry config virtualenvs.in-project true
poetry config --list
# then run install
poetry install

poetry env use $(which python3.10)


### Squash git commits HOWTO
git rev-list --max-parents=0 HEAD