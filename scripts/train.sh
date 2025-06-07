# ENV=server python ../vae/vae_draft_1.py
ENV=server poetry run python ../vae/vae_draft_1.py
runpodctl stop pod $RUNPOD_POD_ID