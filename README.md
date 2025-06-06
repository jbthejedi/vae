# vae
Implementation of a variational autoencoder

# Run before running setup_env.sh to use vim
apt update && apt install vim -y && apt install screen -y

# SCP example
scp -P 42260 -i ~/.ssh/id_ed25519 cifar-10-python.tar.gz root@195.26.233.33:/workspace/data
tar --no-same-owner -xvf cifar-10-python.tar.gz