from typing import Optional
import csv
import os
import subprocess as sp
from typing import Optional
import wandb
import jax
import flax
import optax
import orbax.checkpoint as orbax
from flax import linen as nn
from flax.core import FrozenDict, freeze, frozen_dict
from flax.training import checkpoints, train_state
import numpy as np
from jax import numpy as jnp
from jax.lib import xla_bridge
from functools import partial
from models.pali import (
    cider_score,
    bleu_score,
    combine_metrics,
    compute_weighted_cross_entropy,
    get_loss_normalizing_factor_and_weights,
    SpecialLossNormalizingFactor,
    get_t5x_pretrained,
    get_vit_pretrained,
)


def hardware_setting(
    args,
):
    # Set memory growth for GPU
    set_memory_growth = True
    
    def get_hardware_backend():
        return xla_bridge.get_backend().platform
    
    def get_gpu_memory():
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = (
            sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
        )
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        return memory_free_values
    
    if get_hardware_backend() == "cpu":
        return "cpu"
    elif get_hardware_backend() == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "gpu"
        if set_memory_growth:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        else:
            # set max GPU memory usage
            total_gpu_memory = get_gpu_memory()[args.gpu_id]
            memory_limit = (
                total_gpu_memory * args.gpu_memory_fraction / total_gpu_memory
            )
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_limit) # Allocate GPU memory
        return "gpu"


def init_pali_params(
    config: dict,
    model: nn.Module,
    seed: int = 42,
):
    """ Initialize parameters for the model
    
    Args:
        cfg (dict): config file load from yaml
        model (jax.nn.Module): Pali model
        seed (int, optional): Seed for random number generator. Defaults to 42.
    
    Returns:
        dict: a dictionary of parameters for the Pali model
    """
    
    # Get hyperparameters
    hpparams = config["hyperparams"]
    # Dummy inputs, Pali model requires 4 inputs: 1 for image and 3 for text
    dummy_inputs = {
        "images": jnp.ones(
            shape=(hpparams["train_batch_size"],) + tuple(hpparams["image_size"]) + (3,)
        ),
        "encoder_input_tokens": jnp.ones(
            shape=(hpparams["train_batch_size"], hpparams["encoder_input_tokens"])
        ),
        "decoder_input_tokens": jnp.ones(
            shape=(hpparams["train_batch_size"], hpparams["decoder_input_tokens"])
        ),
        "decoder_target_tokens": jnp.ones(
            shape=(hpparams["train_batch_size"], hpparams["decoder_target_tokens"])
        ),
    }
    # Init parameters
    rng = jax.random.PRNGKey(seed)
    init_rng, init_rng_dropout = jax.random.split(rng, num=2)
    init_rngs = {'params': init_rng, 'dropout': init_rng_dropout}
    variables = model.init(init_rngs, **dummy_inputs, enable_dropout=False) 
    
    return variables


def load_ViT_pretrained(
    config: dict,
    variables: dict,
):
    """ Load pretrained ViT parameters
    
    Args:
        config (dict): config file load from yaml
        variables (dict): a dictionary of parameters for the Pali model
    Returns:
        dict: a dictionary of parameters for the Pali model
    """
    # Load pretrained ViT parameters
    vit_params = get_vit_pretrained(config)
    # Replace ViT parameters in Pali model
    variables = variables.unfreeze()
    variables["params"]["VisionTransformer_0"] = vit_params["params"]
    variables = freeze(variables)
    
    return variables


def load_T5x_pretrained(
    config: dict,
    variables: dict,
):
    """ Load pretrained T5x parameters
    
    Args:
        config (dict): config file load from yaml
        variables (dict): a dictionary of parameters for the Pali model
    Returns:
        dict: a dictionary of parameters for the Pali model
    """
    # Load pretrained ViT parameters
    t5x_params = get_t5x_pretrained(config)
    # Replace ViT parameters in Pali model
    variables = variables.unfreeze()
    variables["params"]["MergeT5_0"] = t5x_params["params"]
    variables = freeze(variables)
    
    return variables


def create_train_state(
    config: dict,
    model: nn.Module,
    variables: dict,
    optimizer_name: str = "adafactor",
):
    """ Create train state for Pali model with ViT parameters is frozen
    
    Args:
        config (dict): config file load from yaml
        model (nn.Module): Pali model
        variables (dict): a dictionary of parameters for the Pali model
        optimizer_name (str, optional): Optimizer name used for training. Defaults to "adafactor".
    
    Returns:
        flax.training.train_state.TrainState: train state for Pali model
    """
    
    # Get hyperparameters
    hpparams = config["hyperparams"]
    # Set training on custom layers by create zero_grads function when performing gradient update
    def zero_grads():
        '''
        Zero out the previous gradient computation
        '''
        def init_fn(_):
            return ()
        def update_fn(updates, state, params=None):
            return jax.tree_map(jnp.zeros_like, updates), ()
        return optax.GradientTransformation(init_fn, update_fn)
    
    # Create the mask for trainable parameters, freeze params is "zero" and trainable params is "<optimizers_name>"
    def create_mask(params, label_fn, optimizer_name="adafactor"):
        def _map(params, mask, label_fn):
            for k in params:
                if label_fn(k):
                    mask[k] = "zero"
                else:
                    if isinstance(params[k], FrozenDict):
                        mask[k] = {}
                        _map(params[k], mask[k], label_fn)
                    else:
                        mask[k] = optimizer_name
        mask = {}
        _map(params, mask, label_fn)
        return frozen_dict.freeze(mask)
    
    # Simple switch case for optimizer
    def get_optimizer(optimizer_name):
        optim = {
            "lion": optax.lion,
            "adamw": optax.adamw,
            "adafactor": optax.adafactor,
        }
        optim = optim.get(optimizer_name, None)
        if optim is None:
            raise ValueError(
                "Not implemented optimizer, the optimizer should be in {}".format(
                    optim.keys()
                )
            )
        return optim
    
    # If you want to freeze some layers, pass it to the lambda function
    # Lambda function should return True if the layer's name is in the list for freezing
    trainable_mask = create_mask(
        variables, lambda x: x in ["VisionTransformer_0"], optimizer_name
    ) # ["VisionTransformer_0", "decoder"]
    
    # Define some optimizer, default is adafactor optimizer
    if optimizer_name == 'lion':
        """ The Lion optimizer.
        
        Lion is discovered by symbolic program search. Unlike most adaptive optimizers
        such as AdamW, Lion only tracks momentum, making it more memory-efficient.
        The update of Lion is produced through the sign operation, resulting in a
        larger norm compared to updates produced by other optimizers such as SGD and
        AdamW. A suitable learning rate for Lion is typically 3-10x smaller than that
        for AdamW, the weight decay for Lion should be in turn 3-10x larger than that
        for AdamW to maintain a similar strength (lr * wd).
        
        References:
            Chen et al, 2023: https://arxiv.org/abs/2302.06675
        """
        tx = optax.chain(
            optax.multi_transform(
                {
                    'lion': get_optimizer(optimizer_name)(
                        learning_rate=hpparams["learning_rate"],
                        weight_decay=1e-2
                    ),
                    'zero': zero_grads(),
                },
                trainable_mask,
            ),
            optax.clip_by_global_norm(hpparams["clipping_threshold"])
        )
    
    elif optimizer_name == 'adafactor':
        """ The Adafactor optimizer.
        
        Adafactor is an adaptive learning rate optimizer that focuses on fast
        training of large scale neural networks. It saves memory by using a factored
        estimate of the second order moments used to scale gradients.
        
        References:
            Shazeer and Stern, 2018: https://arxiv.org/abs/1804.04235
        """
        if hpparams['freeze_vit']:
            tx = optax.multi_transform(
                {
                    'adafactor': get_optimizer(optimizer_name)(
                        learning_rate=hpparams["learning_rate"],
                        decay_rate=0.8,
                        eps=1e-30,
                        decay_offset=0,
                        momentum=0.9,
                        clipping_threshold=hpparams["clipping_threshold"],
                    ),
                    'zero': zero_grads(),
                },
                trainable_mask,
            )
        else:
            tx = optax.adafactor(
                learning_rate=hpparams["learning_rate"],
                decay_rate=0.8,
                eps=1e-30,
                decay_offset=0,
                momentum=0.9,
                clipping_threshold=hpparams["clipping_threshold"],
            )
    
    elif optimizer_name == 'adamw':
        """ Adam with weight decay regularization.
        
        AdamW uses weight decay to regularize learning towards small weights, as
        this leads to better generalization. In SGD you can also use L2 regularization
        to implement this as an additive loss term, however L2 regularization
        does not behave as intended for adaptive gradient algorithms such as Adam.
        
        References:
            Loshchilov et al, 2019: https://arxiv.org/abs/1711.05101
        """
        tx = optax.chain(
            optax.multi_transform(
                {
                    'adamw': get_optimizer(optimizer_name)(
                        learning_rate=hpparams["learning_rate"],
                        b1=0.9,
                        b2=0.999,
                        eps=1e-8,
                        weight_decay=1e-4
                    ),
                    'zero': zero_grads(),
                },
                trainable_mask,
            ),
            optax.clip_by_global_norm(hpparams["clipping_threshold"])
        )
    
    # Create the train state
    return train_state.TrainState.create(apply_fn=model.apply, tx=tx, params=variables)


@jax.jit  # Define the training, evaluating step with @jax.jit for faster training
def _train_step(state: train_state.TrainState, batch, decoder_loss_weights):
    def loss_fn(state, params, batch, decoder_loss_weights):
        dropout_key = jax.random.PRNGKey(seed=0)
        outputs = state.apply_fn(
            params,
            **batch,
            enable_dropout=True,
            rngs={"dropout": dropout_key}
        )
        """ NOTE: When fine-tuning the public T5 checkpoints (trained in T5 MeshTF) the loss 
            normalizing factor should be set to pretraining batch_size * target_token_length.
        """
        # loss_normalizing_factor = batch['decoder_target_tokens'].shape[0] * batch['decoder_target_tokens'].shape[1]
        (loss_normalizing_factor, weights) = get_loss_normalizing_factor_and_weights(
            loss_normalizing_factor=SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS,
            decoder_loss_weights=decoder_loss_weights,
            batch=batch,
        )
        loss, z_loss, weight_sum = compute_weighted_cross_entropy(
            logits=outputs["logits"],
            targets=batch['decoder_target_tokens'],
            weights=weights,
            label_smoothing=0.0,
            z_loss=0.0,
            loss_normalizing_factor=loss_normalizing_factor,
        )
        return loss, outputs["logits"]
    # Create Gradient Function by passing in the function
    grad_fn = jax.value_and_grad(
        loss_fn,
        argnums=1,  # Choose the parameter 'params' in 'loss_fn(state, params, X, y)'
        has_aux=True,  # Return loss and logits for calculating accuracy
    )
    # Calculate the loss and gradients
    (loss, logits), grads = grad_fn(state, state.params, batch, decoder_loss_weights)
    # Update Parameters
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, logits


def train_step(state: train_state.TrainState, batch, decoder_loss_weights):
    """ Perform a training step.
    
    Args:
        state (train_state.TrainState): The current training state.
        batch: Data for the training step.
        decoder_loss_weights: Loss is computed for all but the padding positions.
    
    Returns:
        new_state (train_state.TrainState): The updated training state after the step.
        loss: The calculated loss for the step.
        cider: The CIDEr score calculated based on the logits and target data.
        bleu: The BLEU score calculated based on the logits and target data.
    """
    new_state, loss, logits = _train_step(state, batch, decoder_loss_weights)
    cider, bleu = combine_metrics(logits, batch['decoder_target_tokens'])
    return new_state, loss, cider, bleu


@jax.jit  # Define the training, evaluating step with @jax.jit for faster training
def _eval_step(state: train_state.TrainState, batch, decoder_loss_weights):
    outputs = state.apply_fn(state.params, **batch, enable_dropout=False)
    (loss_normalizing_factor, weights) = get_loss_normalizing_factor_and_weights(
        loss_normalizing_factor=SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS,
        loss_weights=decoder_loss_weights,
        batch=batch,
    )
    loss, z_loss, weight_sum = compute_weighted_cross_entropy(
        logits=outputs["logits"],
        targets=batch['decoder_target_tokens'],
        weights=weights,
        label_smoothing=0.0,
        z_loss=0.0,
        loss_normalizing_factor=loss_normalizing_factor,
    )
    return loss, outputs["logits"]


def eval_step(state: train_state.TrainState, batch, decoder_loss_weights):
    """ Perform an evaluation step.
    
    Args:
        state (train_state.TrainState): The current training state.
        batch: Data for the training step.
        decoder_loss_weights: Loss is computed for all but the padding positions.
    
    Returns:
        loss: The calculated loss for the step.
        cider: The computed CIDEr score.
        bleu: The computed BLEU score.
    """
    loss, logits = _eval_step(state, batch, decoder_loss_weights)
    cider, bleu = combine_metrics(logits, batch['decoder_target_tokens'])
    return loss, cider, bleu


@partial(jax.pmap, axis_name="num_devices")
def train_step_multi_gpu(state: train_state.TrainState, batch, decoder_loss_weights):
    """ Perform a training step on multiple GPUs.
    
    Args:
        state (train_state.TrainState): The current training state.
        batch:  Data for the training step.
        decoder_loss_weights: Loss is computed for all but the padding positions.
    
    Returns:
        new_state (train_state.TrainState): The updated training state after the step.
        loss: The calculated loss for the step.
    """
    @jax.jit
    def loss_fn(state, params, batch, decoder_loss_weights):
        dropout_key = jax.random.PRNGKey(seed=0)
        outputs = state.apply_fn(
            params, 
            **batch,
            enable_dropout=True,
            rngs={"dropout": dropout_key}
        )
        (loss_normalizing_factor, weights) = get_loss_normalizing_factor_and_weights(
            loss_normalizing_factor=SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS,
            loss_weights=decoder_loss_weights,
            batch=batch,
        )
        loss, z_loss, weight_sum = compute_weighted_cross_entropy(
            logits=outputs["logits"],
            targets=batch['decoder_target_tokens'],
            weights=weights,
            label_smoothing=0.0,
            z_loss=0.0,
            loss_normalizing_factor=loss_normalizing_factor,
        )
        return loss, outputs["logits"]
    # Create Gradient Function by passing in the function
    grad_fn = jax.value_and_grad(
        loss_fn,
        argnums=1,  # Choose the parameter 'params' in 'loss_fn(state, params, X, y)'
        has_aux=True,  # Return loss and logits for calculating accuracy
    )
    # Calculate the loss and gradients
    (loss, logits), grads = grad_fn(state, state.params, batch, decoder_loss_weights)
    grads = jax.lax.pmean(grads, axis_name="num_devices")
    loss = jax.lax.pmean(loss, axis_name="num_devices")
    # Update Parameters
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


@partial(jax.pmap, axis_name="num_devices")
def eval_step_multi_gpu(state: train_state.TrainState, batch, decoder_loss_weights):
    """ Perform a multi-GPU evaluation step.
    
    Args:
        state (train_state.TrainState): The current training state.
        batch: Data for the training step.
        decoder_loss_weights: Loss is computed for all but the padding positions.
    
    Returns:
        loss: The calculated loss for the step.
    """
    outputs = state.apply_fn(state.params, **batch, enable_dropout=False)
    (loss_normalizing_factor, weights) = get_loss_normalizing_factor_and_weights(
        loss_normalizing_factor=SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS,
        loss_weights=decoder_loss_weights,
        batch=batch,
    )
    loss, z_loss, weight_sum = compute_weighted_cross_entropy(
        logits=outputs["logits"],
        targets=batch['decoder_target_tokens'],
        weights=weights,
        label_smoothing=0.0,
        z_loss=0.0,
        loss_normalizing_factor=loss_normalizing_factor,
    )
    loss = jax.lax.pmean(loss, axis_name="num_devices")
    return loss


def data_parallel(
    data,
    num_devices: Optional[int] = None,
):
    """ Reshape data for multiple GPUs
    
    Args:
        data: Iterator providing the data for training or evaluation.
        num_devices (int, optional): Number of devices to consider when reshaping the data. Defaults to None.
    
    Returns:
        batch: Data for training
        decoder_loss_weights: Loss is computed for all but the padding positions.
    """
    batch, decoder_loss_weights = next(data)
    batch = {
        "images": batch["images"].reshape(
            [num_devices, -1] + list(batch["images"].shape[1:])
        ),
        "encoder_input_tokens": batch["encoder_input_tokens"].reshape(
            [num_devices, -1] + list(batch["encoder_input_tokens"].shape[1:])
        ),
        "decoder_input_tokens": batch["decoder_input_tokens"].reshape(
            [num_devices, -1] + list(batch["decoder_input_tokens"].shape[1:])
        ),
        "decoder_target_tokens": batch["decoder_target_tokens"].reshape(
            [num_devices, -1] + list(batch["decoder_target_tokens"].shape[1:])
        ),
    }
    decoder_loss_weights = decoder_loss_weights.reshape(
        [num_devices, -1] + list(decoder_loss_weights.shape[1:])
    )
    return batch, decoder_loss_weights


def save_checkpoint_state(
    config: dict,
    state: train_state.TrainState,
):
    """ Save training state and parameters for Pali model
    
    Args:
        config (dict): config file load from yaml
        state (train_state.TrainState): train state for Pali model
    """
    checkpoints_dir = config["checkpoints_dir"]
    os.makedirs(checkpoints_dir["save_state"], exist_ok=True)
    # Save training state
    orbax_checkpointer = orbax.Checkpointer(orbax.PyTreeCheckpointHandler())
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoints_dir["save_state"],
        target=state,
        step=state.step,  # Current step
        overwrite=True,  # Allow to overwrite the old checkpoint
        keep=1,  # Maximum number of checkpoints you want to store
        orbax_checkpointer=orbax_checkpointer,
    )


def save_optimizer(
    config: dict,
    state: train_state.TrainState,
):
    """ Save optimizer state for training to continue training from the saved state
    
    Args:
        config (dict): Configuration file loaded from YAML
        state (train_state.TrainState): Train state containing the optimizer state
    """
    checkpoints_dir = config["checkpoints_dir"]
    os.makedirs(checkpoints_dir["save_opt"], exist_ok=True)
    ckpt_tx = {'opt_state': state.opt_state}
    orbax_checkpointer = orbax.Checkpointer(orbax.PyTreeCheckpointHandler())
    checkpoints.save_checkpoint_multiprocess(
        ckpt_dir=checkpoints_dir["save_opt"],
        target=ckpt_tx,
        step=state.step,
        overwrite=True,
        keep=1,
        orbax_checkpointer=orbax_checkpointer,
    )


def save_history(
    config: dict,
    step: int,
    train_cider: float,
    train_loss: float,
    val_cider: float,
    val_loss: float,
    val_bleu: float,
    train_bleu: float,
    file_name="history.csv",
):
    """ Save training history to csv file
    
    Args:
        config (dict): config file load from yaml
        step (int): the current step when saving the history
        train_cider (float): cider score for training
        train_loss (float): loss value for training
        val_cider (float): cider score for validation
        val_loss (float): loss value for validation
        file_name (str, optional): file name for saving the history. Defaults to "history.csv".
    
    Returns:
        str: path to the saved history file
    """
    checkpoints_dir = config["checkpoints_dir"]
    os.makedirs(checkpoints_dir["save_history"], exist_ok=True)
    file_path = os.path.join(checkpoints_dir["save_history"], file_name)
    with open(file_path, mode="a", newline="") as csv_file:
        fieldnames = ["step", "train_cider", "train_bleu", "train_loss", "val_cider", "val_bleu", "val_loss"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        # write the header row
        if csv_file.tell() == 0:
            writer.writeheader()
        # write the data row
        writer.writerow(
            {
                "step": step,
                "train_cider": train_cider,
                "train_bleu": train_bleu,
                "train_loss": train_loss,
                "val_cider": val_cider,
                "val_bleu": val_bleu,
                "val_loss": val_loss,
            }
        )


def save_history_multi_gpu(
    config: dict,
    step: int,
    train_loss: float,
    val_loss: float,
    val_cider: float,
    val_bleu: float,
    file_name="history_multi_gpu.csv",
):
    """ Save training history of multi-GPU to a CSV file

    Args:
        config (dict): config file load from yaml
        step (int): the current step when saving the history
        train_loss (float): loss value for training
        val_loss (float): loss value for validation
        file_name (str, optional): file name for saving the history. Defaults to "history_multi_gpu.csv".
    
    Returns:
        str: path to the saved history file
    """
    checkpoints_dir = config["checkpoints_dir"]
    os.makedirs(checkpoints_dir["save_history"], exist_ok=True)
    file_path = os.path.join(checkpoints_dir["save_history"], file_name)
    with open(file_path, mode="a", newline="") as csv_file:
        fieldnames = ["step", "train_loss", "val_loss", "val_cider", "val_bleu"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        # write the header row
        if csv_file.tell() == 0:
            writer.writeheader()
        # write the data row
        writer.writerow(
            {
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_cider": val_cider,
                "val_bleu": val_bleu,
            }
        )


def save_parameters(
    config: dict,
    state: train_state.TrainState,
    step: int,
):
    """ Save only the parameters for the model.
    
    Args:
        config (dict): config file load from yaml
        state (train_state.TrainState): train state for model
        step (int): step is currently training
        
    Returns:
        str: path to the saved parameters
    """
    
    checkpoints_dir = config["checkpoints_dir"]
    os.makedirs(checkpoints_dir["save_params"], exist_ok=True)
    params = state.params.unfreeze()
    filename = "params_{}.npz".format(step)
    params_path = os.path.join(checkpoints_dir["save_params"], filename)
    params_arr = np.array(params).reshape(1)
    np.savez_compressed(params_path, params_arr)
    
    return params_path


def init_wandb(
    config: dict,
    project_name: str='pali_model',
    key: str='2340e06838c8262d8e833f86e6d31eb8d76da1f4',
    
):
    hpparams = config["hyperparams"]
    T5 = config["t5"]
    wandb.login(key=key)
    wandb.init(
        project=project_name,
        config={
            "architecture": "Transformer (ViT + T5)",
            "image_size": hpparams["image_size"],
            "encoder_input_tokens": hpparams["encoder_input_tokens"],
            "decoder_input_tokens": hpparams["decoder_input_tokens"],
            "decoder_target_tokens": hpparams["decoder_target_tokens"],
            "vocab_size": T5["vocab_size"],
            "vit_patches_embs_size": T5["vit_patches_embs_size"],
            "train_steps": hpparams["num_steps"],
            "val_steps": hpparams["val_steps"],
            "save_steps": hpparams["save_steps"],
            "train_batch_size": hpparams["train_batch_size"],
            "val_batch_size": hpparams["val_batch_size"],
            "optimizer_name": hpparams["optimizer_name"],
            "learning_rate": hpparams["learning_rate"],
            "clipping_threshold": hpparams["clipping_threshold"],
            "freeze_vit": hpparams["freeze_vit"]
        }
    )