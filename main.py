from datetime import datetime
import logging
from pathlib import Path

from accelerate import Accelerator
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from backbone import MFDiT
from model import MeanFlow
from utils import JVP, Config, Loss, MinMaxNormalizer, SamplerType, TRSampler, cycle

# Set up logging
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
log_filename = LOGS_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to file: {log_filename}")

# Ensure GPU capability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for training.")
device = torch.device("cuda")
logger.info(f"Using PyTorch version: {torch.__version__}")
logger.info(f"Using device: {device}")
logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
logger.info(f"GPU Count: {torch.cuda.device_count()}")
logger.info(f"CUDA Version: {torch.version.cuda}")
logger.info(
    f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
)
logger.info(
    f"GPU Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"
)

accelerator = Accelerator(mixed_precision="fp16")

# Load configuration
config_dict = Config.from_toml("config.toml")
training_config = config_dict["training"]
model_configs = config_dict["model"]

logger.info(f"Training configuration: {training_config}")
logger.info(f"Selected model: {training_config.model_config}")

# Get the selected model configuration
selected_model_config = model_configs[training_config.model_config]
logger.info(f"Model architecture: {selected_model_config}")

# Create data, model, and image directories
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Created data directory: {DATA_DIR}")

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Created model directory: {MODEL_DIR}")

IMAGE_DIR = Path(__file__).parent / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Created image directory: {IMAGE_DIR}")

# Create dataloaders
assert training_config.mode in ["debug", "full"], "Invalid mode"
if training_config.mode == "debug":
    transform = transforms.Compose(
        [
            transforms.Resize((training_config.input_size, training_config.input_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = torchvision.datasets.MNIST(
        root="mnist",
        train=True,
        download=True,
        transform=transform,
    )
    in_channels = 1
    num_classes = 10
elif training_config.mode == "full":
    transform = transforms.Compose(
        [
            transforms.Resize((training_config.input_size, training_config.input_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root="cifar", train=True, download=True, transform=transform
    )
    in_channels = 3
    num_classes = 10

trainloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=training_config.batch_size,
    shuffle=True,
    num_workers=training_config.num_workers,
    drop_last=True,
    pin_memory=True,
)
train_dataloader = cycle(trainloader)
logger.info(f"Created trainloader with {len(trainloader)} batches")

# Create model using selected configuration
model = MFDiT(
    input_size=training_config.input_size,
    patch_size=selected_model_config.patch_size,
    in_channels=in_channels,
    dim=selected_model_config.hidden_dim,
    depth=selected_model_config.depth,
    num_heads=selected_model_config.num_heads,
    num_classes=num_classes,
).to(accelerator.device)
logger.info(
    f"Created model with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters"
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=training_config.learning_rate,
    weight_decay=training_config.weight_decay,
)

# Initialize components for MeanFlow
normalizer = MinMaxNormalizer()
loss_fn = Loss(gamma=training_config.gamma, c=training_config.c)
t_r_sampler = TRSampler(
    sample_ratio=training_config.sample_ratio,
    sampler_type=SamplerType(training_config.sampler_type),
    sample_params=training_config.sample_params,
)
jvp_fn = JVP(use_autograd=False, jvp_config=(1, 0))

# Initialize MeanFlow
meanflow = MeanFlow(
    model=model,
    normalizer=normalizer,
    loss_fn=loss_fn,
    t_r_sampler=t_r_sampler,
    jvp_fn=jvp_fn,
    channels=in_channels,
    image_size=training_config.input_size,
    num_classes=num_classes,
)
logger.info("Initialized MeanFlow training wrapper")

model, optimizer, trainloader = accelerator.prepare(model, optimizer, trainloader)

# Training configuration
n_steps = training_config.n_steps
log_step = 500
sample_step = 5000
checkpoint_step = 5000

logger.info(f"Starting training for {n_steps} steps")
logger.info(f"Batch size: {training_config.batch_size}")
logger.info(f"Learning rate: {training_config.learning_rate}")

# Training loop
global_step = 0
losses = 0.0
mse_losses = 0.0

with tqdm(range(n_steps), dynamic_ncols=True) as pbar:
    pbar.set_description("Training")
    model.train()

    for step in pbar:
        # Get batch
        data = next(train_dataloader)
        x = data[0].to(accelerator.device)
        c = data[1].to(accelerator.device)

        # Forward pass
        loss, mse_val = meanflow.loss(x, c)

        # Backward pass
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        # Update metrics
        global_step += 1
        losses += loss.item()
        mse_losses += mse_val.item()

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "mse": f"{mse_val.item():.4f}"})

        # Logging
        if accelerator.is_main_process and global_step % log_step == 0:
            avg_loss = losses / log_step
            avg_mse = mse_losses / log_step
            lr = optimizer.param_groups[0]["lr"]

            logger.info(
                f"Step {global_step:6d} | Loss: {avg_loss:.6f} | MSE: {avg_mse:.6f} | LR: {lr:.6f}"
            )

            losses = 0.0
            mse_losses = 0.0

        # Generate samples
        if global_step % sample_step == 0 and accelerator.is_main_process:
            model.eval()
            model_module = model.module if hasattr(model, "module") else model

            with torch.no_grad():
                samples = meanflow.sample_each_class(n_per_class=1, sample_steps=5)
                log_img = make_grid(samples, nrow=10)
                img_save_path = IMAGE_DIR / f"step_{global_step}.png"
                save_image(log_img, img_save_path)
                logger.info(f"Saved samples to {img_save_path}")

            accelerator.wait_for_everyone()
            model.train()

        # Save checkpoints
        if global_step % checkpoint_step == 0 and accelerator.is_main_process:
            model_module = model.module if hasattr(model, "module") else model
            ckpt_path = MODEL_DIR / f"step_{global_step}.pt"
            accelerator.save(model_module.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

            accelerator.wait_for_everyone()

# Save final checkpoint
if accelerator.is_main_process:
    model_module = model.module if hasattr(model, "module") else model
    ckpt_path = MODEL_DIR / f"final_step_{global_step}.pt"
    accelerator.save(model_module.state_dict(), ckpt_path)
    logger.info(f"Saved final checkpoint to {ckpt_path}")
logger.info("Training completed!")
