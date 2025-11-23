#!/bin/bash

# Launch distributed training with Accelerate
# This script will automatically use all available GPUs

# Option 1: Use the config file
accelerate launch --config_file accelerate_config.yaml main.py

# Option 2: Specify parameters directly (uncomment to use instead)
# accelerate launch --multi_gpu --num_processes=8 --mixed_precision=bf16 main.py

