import os
import functools

import jax
from jax.lib import xla_bridge
from flax import linen as nn
from jax import numpy as jnp
from t5.data import get_default_vocabulary as get_t5_vocab
from t5x.models import decoding

from .eval_helper import load_checkpoint

OUTPUT_VOCABULARY = get_t5_vocab()

def hardware_setting(args):
    # Set memory growth for GPU
    set_memory_growth = True
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
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
        if set_memory_growth:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
        else:
            # set max GPU memory usage
            total_gpu_memory = get_gpu_memory()[args.gpu_id]
            memory_limit = (
                total_gpu_memory * args.gpu_memory_fraction / total_gpu_memory
            )
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_limit)
            os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
        return "gpu"


# Ref https://github.com/google-research/t5x/blob/main/t5x/models.py#:~:text=cache%27%5D%2C%20initial_index-,def%20predict_batch_with_aux(,%2D1%2C%20%3A%5D%2C%20%7B%27scores%27%3A%20scores%5B%3A%2C%20%2D1%5D%7D,-def%20score_batch(
def _compute_logits_from_slice(
    decoding_state,
    params,
    encoded_inputs,
    raw_inputs,
    max_decode_length,
    pali_model,
):
    """Token slice to logits from decoder model."""
    flat_ids = decoding_state.cur_token
    flat_cache = decoding_state.cache
    
    # flat_ids: [batch * beam, seq_len=1]
    # cache is expanded inside beam_search to become flat_cache
    # flat_cache: [batch * beam, num_heads, depth_per_head, max_decode_len]
    # flat_logits: [batch * beam, seq_len=1, vocab]
    flat_logits, new_vars = pali_model.apply(
        {"params": params["params"], "cache": flat_cache},
        encoded_inputs,
        raw_inputs,  # only needed for encoder padding mask
        flat_ids,
        flat_ids,
        enable_dropout=False,
        decode=True,
        max_decode_length=max_decode_length,
        mutable=["cache"],
        method=pali_model.decode,
    )
    # Remove sequence length dimension since it's always 1 during decoding.
    flat_logits = jnp.squeeze(flat_logits, axis=1)
    new_flat_cache = new_vars["cache"]
    return flat_logits, new_flat_cache


def get_infer_step(
    config: dict,
    model: nn.Module,
    rng: jax.random.KeyArray = None,
    num_decodes: int = 1,
    return_all_decodes: bool = False,
    prompt_with_targets: bool = False,
    batch_size: int = 1,
):
    """Get inference step for Pali model
    
    Args:
        config (dict): config file load from yaml
        model (nn.Module): Pali model
        rng (jax.random.KeyArray, optional): random key. Defaults to None.
        num_decodes (int, optional): the number of beams to use in beam search. Defaults to 1.
        return_all_decodes (bool, optional): whether to return all beams. Defaults to False.
        prompt_with_targets (bool, optional): whether to prompt with targets. Defaults to False.
        batch_size (int, optional): batch size of the model. Defaults to 1.
    Returns:
        function: evaluating step function
    """
    
    hpparams = config["hyperparams"]
    
    # Dummy inputs, Pali model requires 4 inputs: 1 for image and 3 for text
    dummy_inputs = {
        "images": jnp.ones(shape=(batch_size,) + tuple(hpparams["image_size"]) + (3,)),
        "encoder_input_tokens": jnp.ones(
            shape=(batch_size, hpparams["encoder_input_tokens"])
        ),
        "decoder_input_tokens": jnp.ones(
            shape=(batch_size, hpparams["decoder_input_tokens"])
        ),
        "decoder_target_tokens": jnp.ones(
            shape=(batch_size, hpparams["decoder_target_tokens"])
        ),
    }
    
    @jax.jit
    def infer_step(
        params: dict,
        X: dict,
        decoder_params: dict = None,
    ):
        """Predict with fast decoding beam search on a batch.
        
        Here we refer to "parameters" for values that can be compiled into the
        model dynamically, as opposed to static configuration settings that require
        a recompile. For example, the model weights and the decoder brevity-penalty
        are parameters and can be modified without requiring a recompile. The number
        of layers, the batch size and the decoder beam size are configuration
        options that require recompilation if changed.
        
        This method can be used with a customizable decoding function as long as it
        follows the signature of `DecodeFnCallable`. In order to provide a unified
        interface for the decoding functions, we use a generic names. For example, a
        beam size is a concept unique to beam search. Conceptually, it corresponds
        to the number of sequences returned by the beam search.  Therefore, the
        generic argument `num_decodes` corresponds to the beam size if
        `self._decode_fn` is a beam search. For temperature sampling, `num_decodes`
        corresponds to the number of independent sequences to be sampled. Typically
        `num_decodes = 1` is used for temperature sampling.
        
        If `return_all_decodes = True`, the return tuple contains the predictions
        with a shape [batch, num_decodes, max_decode_len] and the scores (i.e., log
        probability of the generated sequence) with a shape [batch, num_decodes].
        
        If `return_all_decodes = False`, the return tuple contains the predictions
        with a shape [batch, max_decode_len] and the scores with a shape [batch].
        
        `decoder_params` can be used to pass dynamic configurations to
        `self.decode_fn`. An example usage is to pass different random seed (i.e.,
        `jax.random.PRNGKey(seed)` with different `seed` value). This can be done by
        setting `decoder_params['decode_rng'] = jax.random.PRNGKey(seed)`.
        
        If `prompt_with_targets = True`, then `decoder_prompt_inputs` is initialized
        from the batch's `decoder_input_tokens`. The EOS is stripped to avoid
        decoding to stop after the prompt by matching to `output_vocabulary.eos_id`.
        Args:
            params: model parameters.
            X: a batch of inputs.
            rng: an optional RNG key to use during prediction, which is passed as
                'decode_rng' to the decoding function.
            decoder_params: additional (model-independent) parameters for the decoder.
            return_all_decodes: whether to return the entire beam or just the top-1.
            num_decodes: the number of beams to use in beam search.
            prompt_with_targets: Whether the force decode decoder_inputs.
        
        Returns:
            A tuple containing:
                the batch of predictions, with the entire beam if requested
                an auxiliary dictionary of decoder scores
        """
        # [batch, input_len]
        inputs_text = X["encoder_input_tokens"]
        target_shape = X["decoder_input_tokens"].shape
        
        # Prepare zeroed-out autoregressive cache.
        outputs, variables_with_cache = model.apply(
            params,
            **dummy_inputs,
            enable_dropout=False,
            decode=True,
            mutable=["cache"],
        )
        
        cache = variables_with_cache["cache"]
        
        # Prepare transformer fast-decoder call for beam search: for beam search, we
        # need to set up our decoder model to handle a batch size equal to
        # batch_size * num_decodes, where each batch item's data is expanded
        # in-place rather than tiled.
        # i.e. if we denote each batch element subtensor as el[n]:
        # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
        # [batch * num_decodes, input_len, emb_dim]
        encoded_inputs = decoding.flat_batch_beam_expand(
            model.apply(
                params,
                **X,
                patch_vit_embs=outputs["patch_embs"],
                enable_dropout=False,
                method=model.encode,
            ),
            num_decodes,
        )
        # Add zeros to the end of the inputs to match the target length.
        inputs_text = jnp.concatenate(
            [inputs_text, jnp.zeros(outputs["patch_embs"].shape[:-1])], axis=-1
        )
        # [batch * num_decodes, input_len]
        raw_inputs = decoding.flat_batch_beam_expand(inputs_text, num_decodes)
        
        tokens_ids_to_logits = functools.partial(
            _compute_logits_from_slice,
            params=params,
            encoded_inputs=encoded_inputs,
            raw_inputs=raw_inputs,
            max_decode_length=target_shape[1],
            pali_model=model,
        )
        if decoder_params is None:
            decoder_params = {}
        
        if rng is not None:
            if decoder_params.get("decode_rng", None) is not None:
                raise ValueError(
                    f"Got RNG both from the `rng` argument ({rng}) and "
                    f"`decoder_params['decode_rng']` ({decoder_params['decode_rng']}). "
                    "Please specify one or the other."
                )
            decoder_params["decode_rng"] = rng
        
        # `decoder_prompt_inputs` is initialized from the batch's
        # `decoder_input_tokens`. The EOS is stripped to avoid decoding to stop
        # after the prompt by matching to `output_vocabulary.eos_id`.
        # These inputs are ignored by the beam search decode fn.
        if prompt_with_targets:
            decoder_prompt_inputs = X["decoder_input_tokens"]
            decoder_prompt_inputs = decoder_prompt_inputs * (
                decoder_prompt_inputs != OUTPUT_VOCABULARY.eos_id
            )
        else:
            decoder_prompt_inputs = jnp.zeros_like(X["decoder_input_tokens"])
        
        # TODO(hwchung): rename the returned value names to more generic ones.
        # Using the above-defined single-step decoder function, run a
        # beam search over possible sequences given input encoding.
        # decodes: [batch, num_decodes, max_decode_len + 1]
        # scores: [batch, num_decodes]
        # hasattr(self.module, 'scan_layers') and self.module.scan_layers
        scanned = False
        
        if "eos_id" not in decoder_params:
            decoder_params["eos_id"] = OUTPUT_VOCABULARY.eos_id
        
        decodes, scores = decoding.beam_search(
            inputs=decoder_prompt_inputs,
            cache=cache,
            tokens_to_logits=tokens_ids_to_logits,
            num_decodes=num_decodes,
            cache_offset=1 if scanned else 0,
            **decoder_params,
        )
        # Beam search returns [n_batch, n_beam, n_length] with beam dimension sorted
        # in increasing order of log-probability.
        # Return the highest scoring beam sequence.
        if return_all_decodes:
            return decodes, {"scores": scores}
        else:
            return decodes[:, -1, :], scores[:, -1]
    
    return infer_step


def decode_output(encode_tokens):
    """Decodes the output tokens to text.
    
    Args:
        encode_tokens (np.array): a batch of encoded tokens or a single encoded tokens.
    
    Returns:
        list: list of decoded text.
    """
    if len(encode_tokens.shape) == 1:
        return [OUTPUT_VOCABULARY.decode(encode_tokens)]
    answers = []
    for tokens in encode_tokens:
        answers.append(OUTPUT_VOCABULARY.decode(tokens))
    return answers