import os
import torch
import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from engine.logger import Logger
from engine.solver import Trainer
from utils.dataloader import (
    build_dataloader,
    build_dataloader_recon,
    build_dataloader_impute,
)
from utils.io_utils import (
    load_yaml_config,
    seed_everything,
    merge_opts_to_config,
    instantiate_from_config,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument("--tensorboard", action="store_true")

    # args for random
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=None)

    # args for training
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--impute", action="store_true")
    parser.add_argument("--reconstruct", action="store_true")
    parser.add_argument("--one_step", action="store_true")
    parser.add_argument("--mode", type=int, default=None)
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--milestone", type=int, default=10)
    parser.add_argument("--missing_ratio", type=float, default=None)

    # args for modify config
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        default=None,
        help="Modify config options using the command-line",
    )

    args = parser.parse_args()
    args.save_dir = os.path.join(args.output, args.name)
    return args


def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)
    config["model"]["params"]["seq_length"] = args.window_size
    config["dataloader"]["train_dataset"]["params"]["window"] = args.window_size
    config["dataloader"]["test_dataset"]["params"]["window"] = args.window_size

    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config["model"]).cuda()
    dataloader_info = build_dataloader(config)
    trainer = Trainer(
        config=config, args=args, model=model, dataloader=dataloader_info, logger=logger
    )

    if args.train:
        trainer.train()

    elif args.one_step:
        trainer.load(args.milestone)
        if args.data_root:
            config["dataloader"]["test_dataset"]["params"]["data_root"] = args.data_root

        if args.mode == 0:  # Train set reconstruction
            config["dataloader"]["test_dataset"]["params"]["period"] = "train"
        elif args.mode == 1:  # Test set reconstruction
            config["dataloader"]["test_dataset"]["params"]["period"] = "test"
        else:
            raise ValueError("mode must be 0 (train set) or 1 (test set).")

        dataloader = build_dataloader_recon(config)["dataloader"]
        losses = trainer.one_step_denoise(dataloader, args.n_trials)

        df = pd.DataFrame(losses, columns=[f"run_{i}" for i in range(args.n_trials)])
        df.to_csv(os.path.join(args.save_dir, f"{args.name}.csv"), index=False)

    elif args.reconstruct:
        trainer.load(args.milestone)
        if args.data_root:
            config["dataloader"]["test_dataset"]["params"]["data_root"] = args.data_root

        if args.mode == 0:  # Train set reconstruction
            config["dataloader"]["test_dataset"]["params"]["period"] = "train"
        elif args.mode == 1:  # Test set reconstruction
            config["dataloader"]["test_dataset"]["params"]["period"] = "test"
        else:
            raise ValueError("mode must be 0 (train set) or 1 (test set).")

        dataloader_info = build_dataloader_recon(config)
        dataloader = dataloader_info["dataloader"]
        dataset = dataloader_info["dataset"]
        sampling_steps = config["dataloader"]["test_dataset"]["sampling_steps"]
        original, samples = trainer.reconstruct(
            dataloader,
            [dataset.window, dataset.var_num],
            sampling_steps,
        )

        losses = ((original - samples) ** 2).reshape(original.shape[0], -1)
        df = pd.DataFrame(losses).astype(np.float32)
        table = pa.Table.from_pandas(df)
        pq.write_table(
            table,
            os.path.join(args.save_dir, f"{args.name}.parquet"),
            compression="zstd",
        )

        # np.savez_compressed(
        #     os.path.join(args.save_dir, args.name),
        #     norm_orig=original,
        #     norm_recon=samples,
        # )

    elif args.impute:
        trainer.load(args.milestone)
        if args.data_root:
            config["dataloader"]["test_dataset"]["params"]["data_root"] = args.data_root

        if args.mode == 0:  # Train set reconstruction
            config["dataloader"]["test_dataset"]["params"]["period"] = "train"
        elif args.mode == 1:  # Test set reconstruction
            config["dataloader"]["test_dataset"]["params"]["period"] = "test"
        else:
            raise ValueError("mode must be 0 (train set) or 1 (test set).")

        dataloader_info = build_dataloader_impute(config, args.missing_ratio)
        dataloader = dataloader_info["dataloader"]
        dataset = dataloader_info["dataset"]
        coef = config["dataloader"]["test_dataset"]["coefficient"]
        stepsize = config["dataloader"]["test_dataset"]["step_size"]
        # sampling_steps = config["dataloader"]["test_dataset"]["sampling_steps"]
        sampling_steps = config["model"]["params"]["sampling_timesteps"]
        original, samples, masks = trainer.impute(
            dataloader,
            [dataset.window, dataset.var_num],
            coef,
            stepsize,
            sampling_steps,
        )

        np.savez_compressed(
            os.path.join(args.save_dir, args.name),
            norm_orig=original,
            norm_impute=samples,
            mask=masks.astype(bool),
        )

    else:
        synthetic_dir = os.path.join("./datasets", "diffusionts")
        os.makedirs(synthetic_dir, exist_ok=True)

        trainer.load(args.milestone)
        dataset = dataloader_info["dataset"]
        samples = trainer.sample(
            num=dataset.total_samples,
            size_every=2000,
            shape=[dataset.window, dataset.var_num],
        )

        np.savez_compressed(
            os.path.join(synthetic_dir, f"{dataset.name}_{dataset.window}"),
            norm_synth=samples,
        )


if __name__ == "__main__":
    main()
