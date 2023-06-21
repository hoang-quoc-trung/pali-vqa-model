import csv
import os
import subprocess as sp
from typing import Optional

import wandb
import jax
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
    cross_entropy_loss,
    compute_weighted_cross_entropy,
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
    """Initialize parameters for the model

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
    variables = model.init(jax.random.PRNGKey(seed), **dummy_inputs) 
    
    return variables


def load_ViT_pretrained(
    config: dict,
    variables: dict,
):
    """Load pretrained ViT parameters

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
    """Load pretrained T5x parameters

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
    """Create train state for Pali model with ViT parameters is frozen

    Args:
        config (dict): config file load from yaml
        model (nn.Module): Pali model
        variables (dict): a dictionary of parameters for the Pali model
        optimizer_name (str, optional): Optimizer name used for training. Defaults to "adam".

    Raises:
        ValueError: Optimizer name is not supported, currently only support "adam" and "sgd"

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
            "sgd": optax.sgd, 
            "lion": optax.lion,
            "adam": optax.adam,
            "adamw": optax.adamw,
            "adagrad": optax.adagrad,
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
        tx = optax.multi_transform(
            {
                'lion': get_optimizer(optimizer_name)(
                    learning_rate=hpparams["learning_rate"],
                    weight_decay=1e-2
                ),
                'zero': zero_grads(),
            },
            trainable_mask,
        )

    elif optimizer_name == 'adafactor':
        """ The Adafactor optimizer.
        
        Adafactor is an adaptive learning rate optimizer that focuses on fast
        training of large scale neural networks. It saves memory by using a factored
        estimate of the second order moments used to scale gradients.

        References:
            Shazeer and Stern, 2018: https://arxiv.org/abs/1804.04235
        """
        tx = optax.multi_transform(
            {
                'adafactor': get_optimizer(optimizer_name)(
                    learning_rate=hpparams["learning_rate"],
                    decay_rate=0.8,
                    eps=1e-30,
                    decay_offset=0
                ),
                'zero': zero_grads(),
            },
            trainable_mask,
        )
        
    elif optimizer_name == 'adagrad':
        """ The Adagrad optimizer.
        
        Adagrad is an algorithm for gradient based optimization that anneals the
        learning rate for each parameter during the course of training.
        
        WARNING: Adagrad's main limit is the monotonic accumulation of squared
        gradients in the denominator: since all terms are >0, the sum keeps growing
        during training and the learning rate eventually becomes vanishingly small.
        
        References:
            Duchi et al, 2011: https://jmlr.org/papers/v12/duchi11a.html
        """
        tx = optax.multi_transform(
            {
                'adagrad': get_optimizer(optimizer_name)(
                    learning_rate=hpparams["learning_rate"],
                    initial_accumulator_value=0.1,
                    eps=1e-7
                ),
                'zero': zero_grads(),
            },
            trainable_mask,
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
        tx = optax.multi_transform(
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
        )
        
    elif optimizer_name == 'sgd':
        """ A canonical Stochastic Gradient Descent optimizer.
        
        This implements stochastic gradient descent. It also includes support for
        momentum, and nesterov acceleration, as these are standard practice when
        using stochastic gradient descent to train deep neural networks.

        References:
            Sutskever et al, 2013: http://proceedings.mlr.press/v28/sutskever13.pdf
        """
        tx = optax.multi_transform(
            {
                'sgd': get_optimizer(optimizer_name)(
                    learning_rate=hpparams["learning_rate"],
                    momentum=None,
                    nesterov=False,
                ),
                'zero': zero_grads(),
            },
            trainable_mask,
        )
        
    else:
        """ The classic Adam optimizer.
        
        Adam is an SGD variant with gradient scaling adaptation. The scaling
        used for each parameter is computed from estimates of first and second-order
        moments of the gradients (using suitable exponential moving averages).
        
        References:
            Kingma et al, 2014: https://arxiv.org/abs/1412.6980
        """
        tx = optax.multi_transform(
            {
                'adam': get_optimizer(optimizer_name)(
                    learning_rate=hpparams["learning_rate"],
                    b1=0.9,
                    b2=0.999,
                    eps=1e-8
                ),
                'zero': zero_grads(),
            },
            trainable_mask,
        )
    
    # tx = optax.chain(
    #     optax.clip_by_global_norm(hpparams["grad_norm_clip"]),
    #     optax.adamw(
    #         learning_rate=hpparams["learning_rate"],
    #     ),
    # )

    # Create the train state
    return train_state.TrainState.create(apply_fn=model.apply, tx=tx, params=variables)


def get_train_val_step(
    config: dict,
):
    """Get training and evaluating step for Pali model

    Args:
        config (dict): config file load from yaml

    Returns:
        function: training step function and evaluating step function
    """
    VOCAB_SIZE = config["t5"]["vocab_size"]

    # Define the training, evaluating step with @jax.jit for faster training
    @jax.jit
    def _train_step(state: train_state.TrainState, X, y):
        def loss_fn(state, params, X, y):
            outputs = state.apply_fn(params, **X)
            logits = outputs["logits"]
            #loss = cross_entropy_loss(logits, y, vocab_size=VOCAB_SIZE)
            
            """ NOTE: When fine-tuning the public T5 checkpoints (trained in T5 MeshTF) the loss normalizing
                        factor should be set to pretraining batch_size * target_token_length.
            """
            loss_normalizing_factor = y.shape[0] * y.shape[1]
            loss = compute_weighted_cross_entropy(
                logits=logits,
                targets=y,
                label_smoothing=0.0,
                z_loss=0.0001,
                loss_normalizing_factor=loss_normalizing_factor,
            )[0]
            return loss, logits

        # Create Gradient Function by passing in the function
        grad_fn = jax.value_and_grad(
            loss_fn,
            argnums=1,  # Choose the parameter 'params' in 'loss_fn(state, params, X, y)'
            has_aux=True,  # Return loss and logits for calculating accuracy
        )
        # Calculate the loss and gradients
        (loss, logits), grads = grad_fn(state, state.params, X, y)
        # Update Parameters
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss, logits

    def train_step(state: train_state.TrainState, X, y):
        new_state, loss, logits = _train_step(state, X, y)
        cider, bleu = combine_metrics(logits, y)
        return new_state, loss, cider, bleu

    @jax.jit
    def _eval_step(state: train_state.TrainState, X, y):
        outputs = state.apply_fn(state.params, **X)
        logits = outputs["logits"]
        # loss = cross_entropy_loss(logits, y, vocab_size=VOCAB_SIZE)
        
        """ NOTE: When fine-tuning the public T5 checkpoints (trained in T5 MeshTF) the loss normalizing
                    factor should be set to pretraining batch_size * target_token_length.
        """
        loss_normalizing_factor = y.shape[0] * y.shape[1]
        loss = compute_weighted_cross_entropy(
            logits=logits,
            targets=y,
            label_smoothing=0.0,
            z_loss=0.0001,
            loss_normalizing_factor=loss_normalizing_factor,
        )[0]
        return loss, logits

    def eval_step(state: train_state.TrainState, X, y):
        loss, logits = _eval_step(state, X, y)
        cider, bleu = combine_metrics(logits, y)
        return loss, cider, bleu

    return train_step, eval_step


def get_train_val_step_multi_gpu(
    config: dict,
):
    """Get training and evaluating step for Pali model - Multi GPU

    Args:
        config (dict): config file load from yaml

    Returns:
        function: training step function and evaluating step function
    """

    @partial(jax.pmap, axis_name="num_devices")
    def _train_step_multi_gpu(state: train_state.TrainState, X, y):
        def loss_fn(state, params, X, y):
            outputs = state.apply_fn(params, **X)
            logits = outputs["logits"]
            """ NOTE: When fine-tuning the public T5 checkpoints (trained in T5 MeshTF) the loss normalizing
                        factor should be set to pretraining batch_size * target_token_length.
            """
            loss_normalizing_factor = y.shape[0] * y.shape[1]
            loss = compute_weighted_cross_entropy(
                logits=logits,
                targets=y,
                label_smoothing=0.0,
                z_loss=0.0001,
                loss_normalizing_factor=loss_normalizing_factor,
            )[0]
            return loss, logits

        # Create Gradient Function by passing in the function
        grad_fn = jax.value_and_grad(
            loss_fn,
            argnums=1,  # Choose the parameter 'params' in 'loss_fn(state, params, X, y)'
            has_aux=True,  # Return loss and logits for calculating accuracy
        )
        # Calculate the loss and gradients
        (loss, logits), grads = grad_fn(state, state.params, X, y)
        grads = jax.lax.pmean(grads, axis_name="num_devices")
        loss = jax.lax.pmean(loss, axis_name="num_devices")
        # Update Parameters
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    def train_step_multi_gpu(state: train_state.TrainState, X, y):
        new_state, loss = _train_step_multi_gpu(state, X, y)
        return new_state, loss[0]

    @partial(jax.pmap, axis_name="num_devices")
    def _eval_step_multi_gpu(state: train_state.TrainState, X, y):
        outputs = state.apply_fn(state.params, **X)
        logits = outputs["logits"]
        """ NOTE: When fine-tuning the public T5 checkpoints (trained in T5 MeshTF) the loss normalizing
                    factor should be set to pretraining batch_size * target_token_length.
        """
        loss_normalizing_factor = y.shape[0] * y.shape[1]
        loss = compute_weighted_cross_entropy(
            logits=logits,
            targets=y,
            label_smoothing=0.0,
            z_loss=0.0001,
            loss_normalizing_factor=loss_normalizing_factor,
        )[0]
        loss = jax.lax.pmean(loss, axis_name="num_devices")
        return loss
    
    def eval_step_multi_gpu(state: train_state.TrainState, X, y):
        loss = _eval_step_multi_gpu(state, X, y)
        return loss[0]
    
    return train_step_multi_gpu, eval_step_multi_gpu


def data_parallel(
    data,
    num_devices: Optional[int] = None,
):
    """Reshape images from [num_devices * batch_size, height, width, channels]
            to [num_devices, batch_size, height, width, img_channels]

    Args:
        data: Iterator providing the data for training or evaluation.
        num_devices (int, optional): Number of devices to consider when reshaping the data. Defaults to None.

    Returns:
        Tuple: A tuple containing the preprocessed input data `X` and the corresponding target data `y`.

    """
    X, y = next(data)
    X = {
        "images": X["images"].reshape(
            [num_devices, -1] + list(X["images"].shape[1:])
        ),
        "encoder_input_tokens": X["encoder_input_tokens"].reshape(
            [num_devices, -1] + list(X["encoder_input_tokens"].shape[1:])
        ),
        "decoder_input_tokens": X["decoder_input_tokens"].reshape(
            [num_devices, -1] + list(X["decoder_input_tokens"].shape[1:])
        ),
        "decoder_target_tokens": X["decoder_target_tokens"].reshape(
            [num_devices, -1] + list(X["decoder_target_tokens"].shape[1:])
        ),
    }
    y = y.reshape([num_devices, -1] + list(y.shape[1:]))
        
    return X, y


def save_checkpoint_state(
    config: dict,
    state: train_state.TrainState,
):
    """Save training state and parameters for Pali model

    Args:
        config (dict): config file load from yaml
        state (train_state.TrainState): train state for Pali model
    """
    hpparams = config["hyperparams"]
    os.makedirs(hpparams["save_checkpoint_dir"], exist_ok=True)
    # Save training state
    # orbax_checkpointer = orbax.Checkpointer(orbax.PyTreeCheckpointHandler())
    checkpoints.save_checkpoint(
        ckpt_dir=hpparams["save_checkpoint_dir"],
        target=state,
        step=state.step,  # Current step
        overwrite=True,  # Allow to overwrite the old checkpoint
        keep=1,  # Maximum number of checkpoints you want to store
        # orbax_checkpointer=orbax_checkpointer,
    )
    
    
def save_checkpoint_state_multi_gpu(
    config: dict,
    state: train_state.TrainState,
):
    """Save training state and parameters for Pali model - Multi GPU

    Args:
        config (dict): config file load from yaml
        state (train_state.TrainState): train state for Pali model
    """
    hpparams = config["hyperparams"]
    os.makedirs(hpparams["save_checkpoint_dir"], exist_ok=True)
    # Save training state
    checkpoints.save_checkpoint_multiprocess(
        ckpt_dir=hpparams["save_checkpoint_dir"],
        target = jax_utils.unreplicate(state),
        step=state.step,  # Current step
        overwrite=True,  # Allow to overwrite the old checkpoint
        keep=1,  # Maximum number of checkpoints you want to store
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
    """Save training history to csv file

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
    hpparams = config["hyperparams"]
    os.makedirs(hpparams["save_history_dir"], exist_ok=True)
    file_path = os.path.join(hpparams["save_history_dir"], file_name)
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
    file_name="history_multi_gpu.csv",
):
    """Save training history of multi-GPU to a CSV file

    Args:
        config (dict): config file load from yaml
        step (int): the current step when saving the history
        train_loss (float): loss value for training
        val_loss (float): loss value for validation
        file_name (str, optional): file name for saving the history. Defaults to "history_multi_gpu.csv".

    Returns:
        str: path to the saved history file
    """
    hpparams = config["hyperparams"]
    os.makedirs(hpparams["save_history_dir"], exist_ok=True)
    file_path = os.path.join(hpparams["save_history_dir"], file_name)
    with open(file_path, mode="a", newline="") as csv_file:
        fieldnames = ["step", "train_loss", "val_loss"]
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
            }
        )
        
        
def save_parameters(
    config: dict,
    state: train_state.TrainState,
    step: int,
):
    """Save only the parameters for the model.

    Args:
        config (dict): config file load from yaml
        state (train_state.TrainState): train state for model
        step (int): step is currently training
        
    Returns:
        str: path to the saved parameters
    """
    
    hpparams = config["hyperparams"]
    os.makedirs(hpparams["save_params_dir"], exist_ok=True)
    params = state.params.unfreeze()
    filename = "params_{}.npz".format(step)
    params_path = os.path.join(hpparams["save_params_dir"], filename)
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
            "learning_rate": hpparams["learning_rate"],
            "grad_norm_clip": hpparams["grad_norm_clip"],
        }
    )