import os
import argparse
import datetime
import logging
import numpy as np
import yaml
from flax.training import checkpoints
from flax.core import freeze
from tqdm import tqdm
import wandb
from models.pali import PaLI
from data.make_dataset import getTFDataGenerator
from utils.train_helper import (
    create_train_state,
    train_step,
    eval_step,
    hardware_setting,
    init_pali_params,
    load_T5x_pretrained,
    load_ViT_pretrained,
    save_checkpoint_state,
    save_history,
    save_parameters,
    init_wandb,
)
from utils.eval_helper import (
    load_checkpoint,
)
from jax.config import config
config.update("jax_debug_nans", True)

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
    device = hardware_setting(args)
    LOGGER.info(f"Training on {device}")
    
    # Load the config file
    cfg = yaml.safe_load(open(args.config_path))
    cfg["checkpoints_dir"]["save_state"] = os.path.join(
        cfg["checkpoints_dir"]["save_state"],
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    hpparams = cfg["hyperparams"]
    checkpoints_dir = cfg["checkpoints_dir"]
    
    # PaLi model
    LOGGER.info("Init PaLI model parameters")
    model = PaLI(cfg)
    variables = init_pali_params(cfg, model, args.seed)
    
    # Load pretrained model
    LOGGER.info("Loading ViT pretrained model")
    variables = load_ViT_pretrained(cfg, variables)
    LOGGER.info("Loading Flan-T5 pretrained model")
    variables = load_T5x_pretrained(cfg, variables)
    
    # Init wandb
    LOGGER.info("Init WanDB")
    init_wandb(cfg)
    
    #----------------------------------------------------------------------------------------
    # # Create the train state
    # state = create_train_state(
    #     cfg,
    #     model,
    #     variables,
    #     optimizer_name="adam",
    # )
    #----------------------------------------------------------------------------------------
    
    # Load params has been trained and continue to train
    if os.path.exists(checkpoints_dir["load_params"]):
        LOGGER.info(
            "Restoring params... from {}".format(checkpoints_dir["load_params"])
        )
        variables = load_checkpoint(checkpoints_dir["load_params"])
        variables = freeze(variables)
        state = create_train_state(
            cfg,
            model,
            variables,
            optimizer_name="adafactor",
        )
    else:
        # Create the train state
        state = create_train_state(
            cfg,
            model,
            variables,
            optimizer_name="adafactor",
        )
    
    # Load state has been trained and continue to train
    if os.path.exists(checkpoints_dir["load_state"]):
        LOGGER.info(
            "Restoring checkpoint... from {}".format(checkpoints_dir["load_state"])
        )
        state = checkpoints.restore_checkpoint(checkpoints_dir["load_state"], state)
    
    # Init dataset
    ds_cfg = cfg["dataset"]
    train_ds, _ = getTFDataGenerator(
        data_root=ds_cfg["training"]["data_root"],
        csv_file=ds_cfg["training"]["csv_file"],
        batch_size=hpparams['train_batch_size'],
        encoder_input_tokens=hpparams['encoder_input_tokens'],
        decoder_target_tokens=hpparams['decoder_target_tokens'],
        shuffle=ds_cfg['shuffle_data'],
        add_eos=ds_cfg['add_eos'],
    )
    val_ds, val_ds_len = getTFDataGenerator(
        data_root=ds_cfg["validation"]["data_root"],
        csv_file=ds_cfg["validation"]["csv_file"],
        batch_size=hpparams['val_batch_size'],
        encoder_input_tokens=hpparams['encoder_input_tokens'],
        decoder_target_tokens=hpparams['decoder_target_tokens'],
        shuffle=ds_cfg['shuffle_data'],
        add_eos=ds_cfg['add_eos'],
    )
    
    def evaluation(state, val_steps):
        val_cider, val_loss, val_bleu = [], [], []
        with tqdm(total=val_steps) as pbar:
            for _ in range(val_steps):
                # Token count exceeds token_length, next data
                while True:
                    try:
                        batch, decoder_loss_weights = next(val_ds_iter)
                        break  
                    except Exception as e:
                        continue
                loss, cider, bleu = eval_step(state, batch, decoder_loss_weights)
                # Save history of metrics across the entire batch
                val_cider.append(cider)
                val_loss.append(loss)
                val_bleu.append(bleu)
                # Update progress bar
                pbar.set_description(
                    f"Validation -> loss: {loss:.3f} - cider: {cider*100:.2f}% - bleu: {bleu*100:.2f}%"
                )
                pbar.update(1)
        return np.mean(val_loss), np.mean(val_cider), np.mean(val_bleu)
    
    LOGGER.info("Start training...")
    """--------------------------------- Start Training Loop ---------------------------------"""
    
    num_steps, save_steps = hpparams["num_steps"], hpparams["save_steps"]
    val_steps = val_ds_len if hpparams["val_steps"] is None else hpparams["val_steps"]
    best_val_cider = 0.0
    # Create iterator for train and validation dataset
    train_ds_iter, val_ds_iter = iter(train_ds), iter(val_ds)
    with tqdm(total=num_steps) as pbar:
        train_cider, train_bleu, train_loss = [], [], []
        for step in range(1, num_steps + 1):
            # Token count exceeds token_length, next data
            while True:
                try:
                    batch, decoder_loss_weights = next(train_ds_iter)
                    break  
                except Exception as e:
                    continue
            state, loss, cider, bleu = train_step(state, batch, decoder_loss_weights)
            # Update progress bar and save history of metrics
            train_cider.append(cider)
            train_loss.append(loss)
            train_bleu.append(bleu)
            pbar.set_description(
                f"Training -> loss: {loss:.3f} - cider: {cider*100:.2f}% - bleu: {bleu*100:.2f}% "
            )
            pbar.update(1)
            
            if step % save_steps == 0:
                avg_train_cider = np.mean(train_cider)
                avg_train_bleu = np.mean(train_bleu)
                avg_train_loss = np.mean(train_loss)
                LOGGER.info(
                    "Mean training in the {} previous steps -> loss: {:.3f} - cider: {:.2f}% - bleu: {:.2f}% ".format(
                        save_steps, avg_train_loss, avg_train_cider*100, avg_train_bleu*100
                    )
                )
                wandb.log({"step": step, "train_loss": avg_train_loss, "train_cider": avg_train_cider*100, "train_bleu": avg_train_bleu*100})
                # Reset the train history
                train_cider, train_bleu, train_loss = [], [], []
                
                LOGGER.info("Performing evaluation...")
                val_loss, val_cider, val_bleu = evaluation(state, val_steps)
                LOGGER.info(
                    "Mean validation in the {} previous steps -> loss: {:.3f} - cider: {:.2f}% - bleu: {:.2f}% ".format(
                        save_steps, val_loss, val_cider * 100, val_bleu * 100
                    )
                )
                wandb.log({"step": step, "val_loss": val_loss, "val_cider": val_cider*100, "val_bleu": val_bleu*100})
                save_history(
                    cfg,
                    step=step,
                    train_cider=avg_train_cider,
                    train_bleu=avg_train_bleu,
                    train_loss=avg_train_loss,
                    val_cider=val_cider,
                    val_bleu=val_bleu,
                    val_loss=val_loss,
                )
                # Save the checkpoint with the highest val_cider_score
                if val_cider > best_val_cider and hpparams["save_best"]:
                    best_val_cider = val_cider
                    # save_checkpoint_state(cfg, state)
                    save_path = save_parameters(cfg, state, step)
                    LOGGER.info("Save the best checkpoint in {}".format(save_path))
                else:
                    # save_checkpoint_state(cfg, state)
                    save_path = save_parameters(cfg, state, step)
                    # LOGGER.info("Save checkpoint in {}".format(save_path))          
    
    # Save the final checkpoint
    # save_checkpoint_state(cfg, state)
    save_path = save_parameters(cfg, state, step)
    LOGGER.info("Training finished! Save the final checkpoint in {}".format(save_path))
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaLI training script")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/default_config.yml",
        help="Path to the config file",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device id")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--gpu_memory_fraction", type=float, default=1.0)
    args = parser.parse_args()
    main(args)