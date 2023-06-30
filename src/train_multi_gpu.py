import os
import argparse
import datetime
import logging
import wandb
import yaml
import jax
import flax
import numpy as np
from flax.training import checkpoints
from flax.core import freeze, unfreeze
from models.pali import PaLI
from tqdm import tqdm
import orbax.checkpoint as orbax
from data.make_dataset import getTFDataGenerator
from utils.train_helper import (
    create_train_state,
    train_step_multi_gpu,
    eval_step,
    hardware_setting,
    init_pali_params,
    load_T5x_pretrained,
    load_ViT_pretrained,
    data_parallel,
    save_history_multi_gpu,
    save_parameters,
    save_checkpoint_state,
    save_optimizer,
    init_wandb,
)
from utils.eval_helper import (
    load_checkpoint,
)

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
    num_devices = jax.device_count()
    device = hardware_setting(args)
    LOGGER.info("Training on: {} {}".format(num_devices, device))

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

    # Create train state
    if os.path.exists(checkpoints_dir["load_params"]):
        # Load params & opt has been trained and continue to train
        LOGGER.info(
            "Restoring params & opt from {}".format(checkpoints_dir["load_params"])
        )
        # Load params
        variables = load_checkpoint(checkpoints_dir["load_params"])
        variables = freeze(variables)
        state = create_train_state(
            cfg,
            model,
            variables,
            optimizer_name=hpparams["optimizer_name"],
        )
        # Load optimizer
        if os.path.exists(checkpoints_dir["load_opt"]):
            state = checkpoints.restore_checkpoint(
                ckpt_dir=checkpoints_dir["load_opt"], 
                target=state.opt,
            )
        
    elif os.path.exists(checkpoints_dir["load_state"]):
        # Load state (model, optimizer, params) has been trained and continue to train
        LOGGER.info(
            "Restoring checkpoint (state) from {}".format(checkpoints_dir["load_state"])
        )
        state = checkpoints.restore_checkpoint(checkpoints_dir["load_state"], state)
        
    else:
        # Create new train state
        LOGGER.info("Create new train state")
        state = create_train_state(
            cfg,
            model,
            variables,
            optimizer_name=hpparams["optimizer_name"],
        )
    state = flax.jax_utils.replicate(state)
    
    # Set batch_size based on number of devices
    train_batch_size = hpparams['train_batch_size']*num_devices
    val_batch_size = hpparams['val_batch_size']*num_devices
    
    # Init dataset
    ds_cfg = cfg["dataset"]
    train_ds, _ = getTFDataGenerator(
        data_root=ds_cfg["training"]["data_root"],
        csv_file=ds_cfg["training"]["csv_file"],
        encoder_input_tokens=hpparams['encoder_input_tokens'],
        decoder_target_tokens=hpparams['decoder_target_tokens'],
        shuffle=ds_cfg['shuffle_data'],
        add_eos=ds_cfg['add_eos'],
        batch_size=train_batch_size,
    )
    val_ds, val_ds_len = getTFDataGenerator(
        data_root=ds_cfg["validation"]["data_root"],
        csv_file=ds_cfg["validation"]["csv_file"],
        encoder_input_tokens=hpparams['encoder_input_tokens'],
        decoder_target_tokens=hpparams['decoder_target_tokens'],
        shuffle=ds_cfg['shuffle_data'],
        add_eos=ds_cfg['add_eos'],
        batch_size=val_batch_size,
    )
    
    def evaluation(state, val_steps):
        val_cider, val_loss, val_bleu = [], [], []
        with tqdm(total=val_steps) as pbar:
            for _ in range(1, val_steps+1):
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
    """----------------------------------------- Start Training Loop -----------------------------------------"""
    
    num_steps, save_steps = hpparams["num_steps"], hpparams["save_steps"]
    val_steps = val_ds_len if hpparams["val_steps"] is None else hpparams["val_steps"]
    best_val_loss = 0.0
    # Create iterator for train and validation dataset
    train_ds_iter, val_ds_iter = iter(train_ds), iter(val_ds)
    with tqdm(total=num_steps) as pbar:
        train_loss = []
        # Train the model
        for step in range(1, num_steps+1):
            # Token count exceeds token_length, next data
            while True:
                try:
                    batch, decoder_loss_weights = data_parallel(
                        data=train_ds_iter, 
                        num_devices=num_devices
                    )
                    break  
                except Exception as e:
                    continue
            state, loss = train_step_multi_gpu(state, batch, decoder_loss_weights)
            loss = float(loss[0])
            # Update progress bar and save history of metrics
            train_loss.append(loss)
            pbar.set_description(f"Training -> loss: {loss:.3f}")
            pbar.update(1)
            
            if step % save_steps == 0:
                avg_train_loss = np.mean(train_loss)
                LOGGER.info( "Mean training in the {} previous steps -> loss: {:.3f}".format(save_steps, avg_train_loss))
                wandb.log({"step": step, "train_loss": avg_train_loss})
                train_loss = [] # Reset the train history
                
                LOGGER.info("Performing evaluation...")
                val_loss, val_cider, val_bleu = evaluation(flax.jax_utils.unreplicate(state), val_steps)
                LOGGER.info(
                    "Mean validation in the {} previous steps -> loss: {:.3f} - cider: {:.2f}% - bleu: {:.2f}% ".format(
                        save_steps, val_loss, val_cider * 100, val_bleu * 100
                    )
                )
                wandb.log({"step": step, "val_loss": val_loss, "val_cider": val_cider*100, "val_bleu": val_bleu*100})
                save_history_multi_gpu(
                    cfg,
                    step=step,
                    train_loss=avg_train_loss,
                    val_cider=val_cider,
                    val_bleu=val_bleu,
                    val_loss=val_loss,
                )
                # Save the checkpoint with the highest val_cider_score
                if val_loss > best_val_loss and hpparams["save_best"]:
                    best_val_loss = val_loss
                    # save_checkpoint_state(cfg, flax.jax_utils.unreplicate(state))
                    save_path = save_parameters(cfg, flax.jax_utils.unreplicate(state), step)
                    LOGGER.info("Save the best checkpoint in {}".format(save_path))
                else:
                    # save_checkpoint_state(cfg, flax.jax_utils.unreplicate(state))
                    save_path = save_parameters(cfg, flax.jax_utils.unreplicate(state), step)
                    LOGGER.info("Save checkpoint in {}".format(save_path))          

    # Save the final checkpoint
    # save_path = save_parameters(cfg, flax.jax_utils.unreplicate(state), step)
    # save_checkpoint_state(cfg, flax.jax_utils.unreplicate(state))
    # LOGGER.info("Training finished! Save the final checkpoint in {}".format(save_path))
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