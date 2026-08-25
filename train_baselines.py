import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Data.weatherbench_data import ERA5


PROJECT_ROOT = Path(__file__).resolve().parent
CLIMODE_DIR = PROJECT_ROOT / "module" / "climode"

MODEL_CHOICES = [
    "phydnet",
    "simvp",
    "tau",
    "weathergft",
    "alphapre",
    "fourcastnet",
    "climode",
    "bfv-ode",
    "all",
]


def parse_args():
    parser = argparse.ArgumentParser("Unified baseline training entry.")
    parser.add_argument("--model", type=str, default="simvp", choices=MODEL_CHOICES)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--smoke_random", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "Data" / "weatherbench")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--train_start", type=str, default="2006-01-01 00:00:00")
    parser.add_argument("--train_end", type=str, default="2015-12-31 23:00:00")
    parser.add_argument("--valid_start", type=str, default="2016-01-01 00:00:00")
    parser.add_argument("--valid_end", type=str, default="2016-12-31 23:00:00")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--output_len", type=int, default=12)
    parser.add_argument("--channels", type=int, default=5)
    parser.add_argument("--img_height", type=int, default=32)
    parser.add_argument("--img_width", type=int, default=64)
    parser.add_argument("--normalization", type=str, default="meanstd", choices=["meanstd", "maxmin", "none"])

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=1)
    parser.add_argument("--max_val_batches", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--tau_reg_weight", type=float, default=0.1)

    parser.add_argument("--simvp_hid_s", type=int, default=64)
    parser.add_argument("--simvp_hid_t", type=int, default=256)
    parser.add_argument("--alphapre_dim", type=int, default=64)
    parser.add_argument("--alphapre_spec_num", type=int, default=8)
    parser.add_argument("--fourcastnet_embed_dim", type=int, default=128)
    parser.add_argument("--fourcastnet_depth", type=int, default=4)
    parser.add_argument("--fourcastnet_patch_size", type=int, default=4)
    parser.add_argument("--fourcastnet_num_blocks", type=int, default=4)

    parser.add_argument("--weathergft_channels", type=int, default=69)
    parser.add_argument("--weathergft_height", type=int, default=128)
    parser.add_argument("--weathergft_width", type=int, default=256)
    parser.add_argument("--weathergft_hidden_dim", type=int, default=64)
    parser.add_argument("--weathergft_use_checkpoint", action="store_true")

    parser.add_argument("--solver", type=str, default="euler")
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--train_lead_steps", type=int, default=2)
    parser.add_argument("--history_len", type=int, default=3)
    parser.add_argument("--low_cutoff", type=float, default=0.35)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limited(loader, max_batches):
    for idx, batch in enumerate(loader):
        if max_batches is not None and max_batches > 0 and idx >= max_batches:
            break
        yield batch


def make_loaders(args, normalization):
    train_set = ERA5(
        data_folder=args.data_dir,
        start_time=args.train_start,
        end_time=args.train_end,
        interval=args.interval,
        input_len=args.input_len,
        output_len=args.output_len,
        normalization=normalization,
        channels=args.channels,
    )
    valid_set = ERA5(
        data_folder=args.data_dir,
        start_time=args.valid_start,
        end_time=args.valid_end,
        interval=args.interval,
        input_len=args.input_len,
        output_len=args.output_len,
        normalization=normalization,
        channels=args.channels,
    )
    return (
        DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True),
        DataLoader(
            valid_set,
            batch_size=args.val_batch_size or args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        ),
    )


def tau_temporal_regularization(pred, target, tau=0.1, eps=1e-12):
    if pred.shape[1] <= 2:
        return pred.new_tensor(0.0)
    pred_gap = (pred[:, 1:] - pred[:, :-1]).flatten(2)
    target_gap = (target[:, 1:] - target[:, :-1]).flatten(2)
    pred_prob = F.softmax(pred_gap / tau, dim=-1)
    target_prob = F.softmax(target_gap / tau, dim=-1)
    return (pred_prob * torch.log(pred_prob / (target_prob + eps) + eps)).mean()


def save_best(model, optimizer, path, epoch, val_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        path,
    )


def build_model(model_name, args, device):
    if model_name in {"simvp", "tau"}:
        from model.simvp import SimVP

        return SimVP(
            shape_in=(args.input_len, args.channels, args.img_height, args.img_width),
            hid_S=args.simvp_hid_s,
            hid_T=args.simvp_hid_t,
            N_S=4,
            N_T=8,
            groups=4,
        ).to(device)

    if model_name == "phydnet":
        from model.phydnet import PhyDNet_Model

        return PhyDNet_Model(
            in_shape=(args.channels, args.img_height, args.img_width),
            T_in=args.input_len,
            T_out=args.output_len,
            device=device,
        ).to(device)

    if model_name == "alphapre":
        from model.alphapre import get_model

        return get_model(
            img_channels=args.channels,
            dim=args.alphapre_dim,
            T_in=args.input_len,
            T_out=args.output_len,
            input_shape=(args.img_height, args.img_width),
            spec_num=args.alphapre_spec_num,
        ).to(device)

    if model_name == "fourcastnet":
        from model.fourcastnet import AFNONet

        params = SimpleNamespace(
            img_shape_x=args.img_height,
            img_shape_y=args.img_width,
            patch_size=args.fourcastnet_patch_size,
            N_in_channels=args.input_len * args.channels,
            N_out_channels=args.output_len * args.channels,
            num_blocks=args.fourcastnet_num_blocks,
        )
        return AFNONet(
            params=params,
            embed_dim=args.fourcastnet_embed_dim,
            depth=args.fourcastnet_depth,
            patch_size=(args.fourcastnet_patch_size, args.fourcastnet_patch_size),
        ).to(device)

    if model_name == "weathergft":
        from model.weather_gft import GFT

        if args.weathergft_height != 128 or args.weathergft_width != 256:
            raise ValueError("WeatherGFT uses a fixed 128x256 input grid in this package.")
        return GFT(
            hidden_dim=args.weathergft_hidden_dim,
            channels=args.weathergft_channels,
            use_checkpoint=args.weathergft_use_checkpoint,
        ).to(device)

    raise ValueError(model_name)


def forward_loss(model_name, model, x, y, args):
    if model_name in {"simvp", "tau"}:
        pred = model(x)
        target = y[:, : pred.shape[1]]
        loss = F.mse_loss(pred, target)
        if model_name == "tau":
            loss = loss + args.tau_reg_weight * tau_temporal_regularization(pred, target)
        return pred, loss

    if model_name == "phydnet":
        return model.predict(x, y, compute_loss=True)

    if model_name == "alphapre":
        pred, loss_dict = model.predict(x, y, compute_loss=True)
        return pred, loss_dict["total_loss"]

    if model_name == "fourcastnet":
        b, _, c, h, w = x.shape
        pred = model(x.reshape(b, -1, h, w)).reshape(b, args.output_len, c, h, w)
        return pred, F.mse_loss(pred, y)

    if model_name == "weathergft":
        frame = x[:, -1]
        pred = model(frame)
        target = torch.zeros_like(pred)
        if pred.ndim == 5 and y.shape[1] >= pred.shape[1] and y.shape[2] == pred.shape[2]:
            target = y[:, : pred.shape[1]]
        elif pred.ndim == 4 and y.shape[2] == pred.shape[1]:
            target = y[:, 0]
        return pred, F.mse_loss(pred, target)

    raise ValueError(model_name)


@torch.no_grad()
def validate(model_name, model, loader, device, args):
    model.eval()
    total = 0.0
    count = 0
    for x, y in limited(loader, args.max_val_batches):
        x, y = x.to(device), y.to(device)
        _pred, loss = forward_loss(model_name, model, x, y, args)
        total += float(loss.detach().cpu())
        count += 1
    return total / max(count, 1)


def random_batch(model_name, args, device):
    channels = args.weathergft_channels if model_name == "weathergft" else args.channels
    height = args.weathergft_height if model_name == "weathergft" else args.img_height
    width = args.weathergft_width if model_name == "weathergft" else args.img_width
    x = torch.randn(args.batch_size, args.input_len, channels, height, width, device=device)
    y = torch.randn(args.batch_size, args.output_len, channels, height, width, device=device)
    return x, y


def train_internal(model_name, args):
    device = torch.device(args.device)
    model = build_model(model_name, args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_path = args.output_dir / model_name / f"best_{model_name}.pth"
    best_val = float("inf")

    train_loader = valid_loader = None
    if not args.smoke_random:
        norm = "maxmin" if model_name == "phydnet" else args.normalization
        train_loader, valid_loader = make_loaders(args, norm)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        batches = [random_batch(model_name, args, device)] if args.smoke_random else limited(train_loader, args.max_train_batches)
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            _pred, loss = forward_loss(model_name, model, x, y, args)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_loss = float(np.mean(losses)) if args.smoke_random else validate(model_name, model, valid_loader, device, args)
        print(f"{model_name} epoch {epoch + 1}/{args.epochs} train={np.mean(losses):.6f} val={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            save_best(model, optimizer, best_path, epoch + 1, val_loss)
            print("saved", best_path)


def climode_command(args, bfv=False):
    script = "train_global_freq_v2.py" if bfv else "train_global_v2.py"
    out_name = "bfv-ode" if bfv else "climode"
    command = [
        sys.executable,
        script,
        "--solver",
        args.solver,
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
        "--niters",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--train_lead_steps",
        str(args.train_lead_steps),
        "--device",
        args.device,
        "--output_dir",
        str(args.output_dir / out_name),
        "--data_dir",
        str(args.data_dir / "era5_data"),
        "--velocity_dir",
        str(args.data_dir / "velocity"),
        "--max_train_batches",
        str(args.max_train_batches),
        "--max_val_batches",
        str(args.max_val_batches),
    ]
    if bfv:
        command += ["--history_len", str(args.history_len), "--low_cutoff", str(args.low_cutoff)]
    return command


def run_delegated(model_name, args):
    command = climode_command(args, bfv=(model_name == "bfv-ode"))
    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("Working directory:", CLIMODE_DIR)
    print("Command:", " ".join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=str(CLIMODE_DIR), env=env, check=True)


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print("Dry run enabled.")

    models = MODEL_CHOICES[:-1] if args.model == "all" else [args.model]
    for model_name in models:
        print("=" * 80)
        print("Model:", model_name)
        start = time.time()
        if model_name in {"climode", "bfv-ode"}:
            run_delegated(model_name, args)
        elif args.dry_run:
            print(f"[dry-run] internal trainer selected for {model_name}")
        else:
            train_internal(model_name, args)
        print(f"Finished {model_name} in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
