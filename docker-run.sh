#!/bin/bash

# Run training in Docker container (single GPU)
echo "Starting training container on GPU 0..."

docker run --gpus '"device=0"' \
    --name meanflow-training \
    --rm \
    -v "$(pwd)/logs:/workspace/logs" \
    -v "$(pwd)/models:/workspace/models" \
    -v "$(pwd)/images:/workspace/images" \
    -v "$(pwd)/runs:/workspace/runs" \
    -v "$(pwd)/data:/workspace/data" \
    -v "$(pwd)/mnist:/workspace/mnist" \
    -v "$(pwd)/cifar:/workspace/cifar" \
    -p 6006:6006 \
    meanflow-training:latest

echo "Training completed!"
