import jax
import numpy as np
from flax import linen as nn
from .train_helper import (
    hardware_setting,
    get_loss_normalizing_factor_and_weights,
    compute_weighted_cross_entropy,
    SpecialLossNormalizingFactor,
    
)
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


def get_eval_step(model: nn.Module):
    """Get evaluating step for Pali model

    Args:
        model (nn.Module): Pali model

    Returns:
        function: evaluating step function
    """

    # Enable jit for faster evaluation
    @jax.jit
    def val_step(params, batch, decoder_loss_weights):
        outputs = model.apply(params, **batch)
        (loss_normalizing_factor, weights) = get_loss_normalizing_factor_and_weights(
            loss_normalizing_factor=SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS,
            loss_weights=decoder_loss_weights,
            batch=batch,
        )
        loss, z_loss, weight_sum = compute_weighted_cross_entropy(
            logits=outputs["logits"],
            targets=batch['decoder_target_tokens'],
            weights=weights,
            label_smoothing=0.1,
            z_loss=0.0001,
            loss_normalizing_factor=loss_normalizing_factor,
        )
        return loss, outputs["logits"]

    # Currently, CIDEr metric is not supported in jax.jit
    def eval_step(params, batch, decoder_loss_weights):
        loss, logits = val_step(params, batch, decoder_loss_weights)
        cider, bleu = combine_metrics(logits, batch['decoder_target_tokens'])
        return loss, cider, bleu

    return eval_step
