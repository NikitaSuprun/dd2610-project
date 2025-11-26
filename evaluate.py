import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from torchmetrics.image.fid import FrechetInceptionDistance

# project imports
from backbone import MFDiT
from model import MeanFlow
from utils import JVP, Config, Loss, MinMaxNormalizer, SamplerType, TRSampler


@torch.no_grad()
def compute_fid_for_checkpoint(
    ckpt_path: str,
    config_path: str = "config.toml",
) -> float:
    """
    Compute FID for a trained MeanFlow checkpoint on CIFAR-10 or MNIST *in memory*.

    Evaluation settings (num_gen, gen_batch_per_class, sample_steps, dataset) are read from
    the [evaluation] section in config.toml.

    Args:
        ckpt_path: Path to model .pt file
        config_path: Path to your config.toml

    Returns:
        FID score (float)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(ckpt_path)

    # Configs
    config_dict = Config.from_toml(config_path)
    training_config = config_dict["training"]
    model_configs = config_dict["model"]
    selected_model_config = model_configs[training_config.model_config]
    eval_config = config_dict.get("evaluation", {})

    # Load evaluation settings from config with sensible defaults
    dataset = eval_config.get("dataset", "cifar10").lower()
    num_gen = eval_config.get("num_gen", 50_000)
    gen_batch_per_class = eval_config.get("gen_batch_per_class", 500)
    sample_steps = eval_config.get("sample_steps", 5)

    # Validate dataset
    if dataset not in ("cifar10", "mnist"):
        raise ValueError(f"dataset must be 'cifar10' or 'mnist', got '{dataset}'")


    # Dataset-specific settings
    if dataset == "cifar10":
        if training_config.mode != "full":
            raise ValueError("FID eval on CIFAR10 is meant for mode='full'.")
        in_channels = 3
        num_classes = 10
    else:  # mnist
        if training_config.mode != "mnist":
            raise ValueError("FID eval on MNIST is meant for mode='mnist'.")
        in_channels = 1
        num_classes = 10

    image_size = training_config.input_size

    assert num_gen % num_classes == 0, f"num_gen must be divisible by num_classes ({num_classes})."
    n_per_class_total = num_gen // num_classes

    if sample_steps is None:
        sample_steps = training_config.sampling_steps

    # Load dataset
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # [0,1]
        ]
    )

    if dataset == "cifar10":
        test_dataset = torchvision.datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=transform,
        )
        dataset_name = "CIFAR-10"
    else:  # mnist
        test_dataset = torchvision.datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=transform,
        )
        dataset_name = "MNIST"

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


    # Build model
    model = MFDiT(
        condition=training_config.condition_type,
        input_size=image_size,
        patch_size=selected_model_config.patch_size,
        in_channels=in_channels,
        dim=selected_model_config.hidden_dim,
        depth=selected_model_config.depth,
        num_heads=selected_model_config.num_heads,
        num_classes=num_classes,
        use_checkpoint=training_config.use_gradient_checkpointing,
    ).to(device)

    normalizer = MinMaxNormalizer()
    loss_fn = Loss(gamma=training_config.gamma, c=training_config.c)
    t_r_sampler = TRSampler(
        sample_ratio=training_config.sample_ratio,
        sampler_type=SamplerType(training_config.sampler_type),
        sample_params=training_config.sample_params,
    )
    jvp_fn = JVP(use_autograd=training_config.jvp_use_autograd, jvp_config=(1, 0))

    meanflow = MeanFlow(
        model=model,
        normalizer=normalizer,
        loss_fn=loss_fn,
        t_r_sampler=t_r_sampler,
        jvp_fn=jvp_fn,
        channels=in_channels,
        image_size=image_size,
        num_classes=num_classes,
        cfg_ratio=training_config.cfg_ratio,
        cfg_scale=training_config.cfg_scale,
    )

    # Checkpoint
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Fid metric
    # normalize=True -> expects inputs in [0, 1], scales internally to [0, 255]
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True)
    fid_metric = fid_metric.to(device)



    print(f"Feeding real {dataset_name} test images to FID metric...")
    for x_real, _ in test_loader:
        x_real = x_real.to(device)
        fid_metric.update(x_real, real=True)

    # Generate images
    print(
        f"Generating {num_gen} images total "
        f"({n_per_class_total} per class) in memory..."
    )

    gen_batch_per_class = max(1, gen_batch_per_class)
    iters = (n_per_class_total + gen_batch_per_class - 1) // gen_batch_per_class

    generated = 0
    pbar = tqdm(range(iters), desc="Sampling", unit="iter")
    for it in pbar:
        remaining_per_class = n_per_class_total - it * gen_batch_per_class
        if remaining_per_class <= 0:
            break
        current_n_per_class = min(gen_batch_per_class, remaining_per_class)

        pbar.set_postfix({
            "iter": f"{it+1}/{iters}",
            "per_class": current_n_per_class,
            "images_this_iter": current_n_per_class * num_classes,
            "generated": generated,
        })

        samples = meanflow.sample_each_class(
            n_per_class=current_n_per_class,
            sample_steps=sample_steps,
            )

        samples = samples.to(device)
        fid_metric.update(samples, real=False)
        generated += samples.size(0)
        pbar.set_postfix({"generated": generated})
    pbar.close()

    print(f"Total generated samples fed to FID metric: {generated}")

    # Compute FID
    fid_value = fid_metric.compute().item()
    print(f"FID ({dataset_name} test) for {ckpt_path.name}: {fid_value:.4f}")
    return fid_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str, help="Path to .pt checkpoint")
    args = parser.parse_args()

    compute_fid_for_checkpoint(
        ckpt_path=args.ckpt_path,
        config_path="config.toml",
    )
