# Run the training script
ENV=server poetry run python /workspace/vae/vae/vae_conv.py

# Capture the exit code
EXIT_CODE=$?

# If the script succeeded, stop the pod
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Script succeeded. Shutting down pod..."
    runpodctl stop pod $RUNPOD_POD_ID
else
    echo "❌ Script failed with exit code $EXIT_CODE. Pod will remain running."
fi