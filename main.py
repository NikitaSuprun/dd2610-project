from datetime import datetime
import logging
from pathlib import Path

from accelerate import Accelerator
import torch
from torch.utils.tensorboard import SummaryWriter
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

# Set up TensorBoard
tensorboard_dir = (
    Path(__file__).parent
    / "runs"
    / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
writer = SummaryWriter(log_dir=tensorboard_dir)
logger.info(f"TensorBoard logging to: {tensorboard_dir}")
logger.info("Start TensorBoard with: tensorboard --logdir=runs")

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

# Load configuration
config_dict = Config.from_toml("config.toml")
training_config = config_dict["training"]
model_configs = config_dict["model"]

# Initialize accelerator with config
accelerator = Accelerator(mixed_precision=training_config.mixed_precision)
logger.info(
    f"Initialized Accelerator with mixed_precision={training_config.mixed_precision}"
)

logger.info(f"Training configuration: {training_config}")
logger.info(f"Selected model: {training_config.model_config}")

# Get the selected model configuration
selected_model_config = model_configs[training_config.model_config]
logger.info(f"Model architecture: {selected_model_config}")

# Log hyperparameters to TensorBoard
hparams = {
    "model": training_config.model_config,
    "mode": training_config.mode,
    "batch_size": training_config.batch_size,
    "learning_rate": training_config.learning_rate,
    "weight_decay": training_config.weight_decay,
    "n_epochs": training_config.n_epochs,
    "depth": selected_model_config.depth,
    "hidden_dim": selected_model_config.hidden_dim,
    "num_heads": selected_model_config.num_heads,
    "patch_size": selected_model_config.patch_size,
    "gamma": training_config.gamma,
    "c": training_config.c,
    "sample_ratio": training_config.sample_ratio,
    "cfg_ratio": training_config.cfg_ratio,
    "cfg_scale": training_config.cfg_scale,
}

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
        root=DATA_DIR,
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
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    in_channels = 3
    num_classes = 10

trainloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=training_config.batch_size,
    shuffle=True,
    num_workers=training_config.num_workers,
    drop_last=True,
    pin_memory=False,
)
logger.info(f"Created trainloader with {len(trainloader)} batches per epoch")
logger.info(f"Dataset size: {len(dataset)} samples")

# Create model using selected configuration
model = MFDiT(
    condition=training_config.condition_type,
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
jvp_fn = JVP(use_autograd=training_config.jvp_use_autograd, jvp_config=(1, 0))

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
    cfg_ratio=training_config.cfg_ratio,
    cfg_scale=training_config.cfg_scale,
)
logger.info("Initialized MeanFlow training wrapper")

model, optimizer, trainloader = accelerator.prepare(model, optimizer, trainloader)

# Log model graph to TensorBoard
try:
    # Create sample input for model graph
    sample_batch = next(iter(trainloader))
    sample_x = sample_batch[0][:1].to(accelerator.device)  # Single sample
    sample_t = torch.rand(1, device=accelerator.device)
    sample_r = torch.rand(1, device=accelerator.device)
    sample_c = torch.zeros(1, dtype=torch.long, device=accelerator.device)

    writer.add_graph(model, (sample_x, sample_t, sample_r, sample_c))
    logger.info("Model graph logged to TensorBoard")
except Exception as e:
    logger.error(f"Could not log model graph: {e}")
    # Fail the training, as we do not want to continue with a large training without observability
    raise e

# Training configuration
n_epochs = training_config.n_epochs
log_step = training_config.log_step
sample_every_n_epochs = training_config.sample_every_n_epochs
checkpoint_every_n_epochs = training_config.checkpoint_every_n_epochs
histogram_step = training_config.histogram_step

steps_per_epoch = len(trainloader)
total_steps = n_epochs * steps_per_epoch

logger.info(f"Starting training for {n_epochs} epochs ({total_steps} total steps)")
logger.info(f"Steps per epoch: {steps_per_epoch}")
logger.info(f"Batch size: {training_config.batch_size}")
logger.info(f"Learning rate: {training_config.learning_rate}")
logger.info(
    f"Log every {log_step} steps | Samples every {sample_every_n_epochs} epochs | Checkpoints every {checkpoint_every_n_epochs} epochs"
)

# Training loop
global_step = 0
losses = 0.0
mse_losses = 0.0

for epoch in range(n_epochs):
    logger.info(f"Starting Epoch {epoch + 1}/{n_epochs}")
    model.train()
    
    with tqdm(trainloader, dynamic_ncols=True) as pbar:
        pbar.set_description(f"Epoch {epoch + 1}/{n_epochs}")
        
        for data in pbar:
            # Get batch
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
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "mse": f"{mse_val.item():.4f}",
                "epoch": f"{epoch + 1}/{n_epochs}"
            })

            # Logging
            if accelerator.is_main_process and global_step % log_step == 0:
                avg_loss = losses / log_step
                avg_mse = mse_losses / log_step
                lr = optimizer.param_groups[0]["lr"]

                logger.info(
                    f"Epoch {epoch + 1}/{n_epochs} | Step {global_step:6d} | Loss: {avg_loss:.6f} | MSE: {avg_mse:.6f} | LR: {lr:.6f}"
                )

                # Log to TensorBoard
                writer.add_scalar("Loss/train", avg_loss, global_step)
                writer.add_scalar("Loss/mse", avg_mse, global_step)
                writer.add_scalar("Learning_Rate", lr, global_step)
                writer.add_scalar("Epoch", epoch + 1, global_step)

                # Log model parameters and gradients histograms
                if global_step % histogram_step == 0:
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            writer.add_histogram(
                                f"Parameters/{name}", param.data, global_step
                            )
                            writer.add_histogram(
                                f"Gradients/{name}", param.grad.data, global_step
                            )

                losses = 0.0
                mse_losses = 0.0
    
    # End of epoch operations
    if accelerator.is_main_process:
        # Generate samples at end of epoch
        if (epoch + 1) % sample_every_n_epochs == 0:
            logger.info(f"Generating samples at end of epoch {epoch + 1}")
            model.eval()
            model_module = model.module if hasattr(model, "module") else model

            with torch.no_grad():
                samples = meanflow.sample_each_class(
                    n_per_class=training_config.n_samples_per_class,
                    sample_steps=training_config.sampling_steps,
                )
                log_img = make_grid(samples, nrow=training_config.sample_grid_nrow)
                img_save_path = IMAGE_DIR / f"epoch_{epoch + 1}.png"
                save_image(log_img, img_save_path)
                logger.info(f"Saved samples to {img_save_path}")

                # Log images to TensorBoard
                writer.add_image("Generated_Samples", log_img, global_step)

            accelerator.wait_for_everyone()
            model.train()

        # Save checkpoints at end of epoch
        if (epoch + 1) % checkpoint_every_n_epochs == 0:
            logger.info(f"Saving checkpoint at end of epoch {epoch + 1}")
            model_module = model.module if hasattr(model, "module") else model
            ckpt_path = MODEL_DIR / f"epoch_{epoch + 1}.pt"
            accelerator.save(model_module.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

            accelerator.wait_for_everyone()

# Save final checkpoint
if accelerator.is_main_process:
    model_module = model.module if hasattr(model, "module") else model
    ckpt_path = MODEL_DIR / f"final_epoch_{n_epochs}.pt"
    accelerator.save(model_module.state_dict(), ckpt_path)
    logger.info(f"Saved final checkpoint to {ckpt_path}")

    # Log final metrics and hyperparameters
    final_metrics = {
        "final_loss": avg_loss if "avg_loss" in locals() else 0.0,
        "final_mse": avg_mse if "avg_mse" in locals() else 0.0,
        "total_steps": global_step,
        "total_epochs": n_epochs,
    }
    writer.add_hparams(hparams, final_metrics)

    # Close TensorBoard writer
    writer.close()
    logger.info("TensorBoard writer closed")

logger.info(f"Training completed! Trained for {n_epochs} epochs ({global_step} steps)")
logger.info("View results with: tensorboard --logdir=runs")
