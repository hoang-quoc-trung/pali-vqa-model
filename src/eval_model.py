import argparse
import logging
import os

import jax
import numpy as np
import yaml
from data.make_dataset import getTFDataGenerator
from models.pali import PaLI
from tqdm import tqdm
from utils.eval_helper import get_eval_step, hardware_setting, load_checkpoint

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def main(args):
    # Testcase
    assert os.path.exists(
        args.config_path
    ), f"Config file {args.config_path} does not exist!"
    # Hardware setting
    hardware_setting(args)
    # Load the config file
    cfg = yaml.safe_load(open(args.config_path))
    hpparams = cfg["hyperparams"]

    # Init model
    model = PaLI(cfg)

    if not os.path.exists(args.ckpt):
        raise ValueError(f"Checkpoint file {args.ckpt} does not exist!")
    LOGGER.info(f"Loading PaLI checkpoint from {args.ckpt}")
    params = load_checkpoint(args.ckpt)

    ds_cfg = cfg["dataset"]
    val_ds, val_ds_len = getTFDataGenerator(
        data_root=ds_cfg["validation"]["data_root"],
        csv_file=ds_cfg["validation"]["csv_file"],
        batch_size=args.batch_size,
    )

    eval_step_fn = get_eval_step(cfg, model)

    LOGGER.info("Start evaluation on validation dataset")
    val_ds_iter = iter(val_ds)
    val_steps = val_ds_len // args.batch_size
    # List stores the history of metrics
    batch_cider, batch_loss = [], []
    with tqdm(total=val_steps) as pbar:
        # Evaluate the model on 5000 samples
        for _ in range(val_steps):
            X, y = next(val_ds_iter)
            loss, cider = eval_step_fn(params, X, y)
            # Save history of metrics across the entire batch
            batch_cider.append(cider)
            batch_loss.append(loss)
            # Update progress bar
            pbar.set_description(
                f"Validation -> loss: {loss:.3f} - cider: {cider*100:.2f}% "
            )
            pbar.update(1)
    val_loss, val_cider = np.mean(batch_loss), np.mean(batch_cider)
    LOGGER.info(
        f"Mean validation -> loss: {val_loss:.3f} - cider: {val_cider*100:.2f}% "
    )
    with open("validation.txt", "a") as f:
        f.write(
            f"Peform validation on {hpparams.get('load_checkpoint_dir')} with dataset \n{ds_cfg['validation']['data_root']}, {ds_cfg['validation']['csv_file']}\n The length of validation dataset is {val_ds_len}"
        )
        f.write(
            f"Mean validation -> loss: {val_loss:.3f} - cider: {val_cider*100:.2f}% "
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaLI training script")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/default_config_local.yml",
        help="Path to the config file",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device id")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to the checkpoint file. (.npy)"
    )
    args = parser.parse_args()
    main(args)
