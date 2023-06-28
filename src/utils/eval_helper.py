import jax
import numpy as np
from flax import linen as nn
from models.pali import cider_score, cross_entropy_loss

from .train_helper import hardware_setting
from models.pali import combine_metrics


def load_checkpoint(checkpoints_path: str):
    """Load only the parameters for the model.

    Args:
        checkpoint_dir (str): path to load the model's parameters

    Returns:
        params: parameters for the model
    """
    params = np.load(checkpoints_path, allow_pickle=True)
    params = params['arr_0'][0]
    return params


def get_eval_step(config: dict, model: nn.Module):
    """Get evaluating step for Pali model

    Args:
        config (dict): config file load from yaml

    Returns:
        function: evaluating step function
    """
    VOCAB_SIZE = config["t5"]["vocab_size"]

    # Enable jit for faster evaluation
    @jax.jit
    def val_step(params, X, y):
        outputs = model.apply(params, **X)
        logits = outputs["logits"]
        loss = cross_entropy_loss(logits, y, vocab_size=VOCAB_SIZE)
        return loss, logits

    # Currently, CIDEr metric is not supported in jax.jit
    def eval_step(params, X, y):
        loss, logits = val_step(params, X, y)
        cider, bleu = combine_metrics(logits, y)
        return loss, cider, bleu

    return eval_step
