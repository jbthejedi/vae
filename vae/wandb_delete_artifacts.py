import wandb

api = wandb.Api()
artifact_versions = api.artifact_versions("model", "jbarry-team/vae-cifar10/vae-conv-orig_best_model")

# Keep the most recent (assumed to be the first in the list)
for artifact in list(artifact_versions)[3:]:
    artifact.delete()
