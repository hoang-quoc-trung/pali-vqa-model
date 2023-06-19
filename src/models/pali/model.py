import gin
import jax
import jax.numpy as jnp
import nest_asyncio
from flax import linen as nn
from flax.core import freeze, unfreeze
from jax.random import PRNGKey
from models.vit.vit_jax import checkpoint
from models.vit.vit_jax.configs import models as models_vit_config
from t5x.checkpoints import load_t5x_checkpoint
from t5x.examples.t5.network import Decoder, EncoderLayer, T5Config
from t5x.examples.t5.network import Transformer as T5
from t5x.examples.t5.network import layers
from vit_jax.models import VisionTransformer as ViT


class Encoder(nn.Module):
    """A stack of encoder layers.
    Ref:src/models/t5x/t5x/examples/t5/network.py line 174
    """

    config: T5Config
    shared_embedding: nn.Module

    @nn.compact
    def __call__(
        self,
        encoder_input_tokens,
        patch_vit_embs,
        encoder_mask=None,
        deterministic=False,
    ):
        cfg = self.config
        assert encoder_input_tokens.ndim == 2  # [batch, length]
        rel_emb = layers.RelativePositionBiases(
            num_buckets=32,
            max_distance=128,
            num_heads=cfg.num_heads,
            dtype=cfg.dtype,
            embedding_init=nn.initializers.variance_scaling(1.0, "fan_avg", "uniform"),
            name="relpos_bias",
        )
        # [batch, length] -> [batch, length, emb_dim]
        # Input [batch, 38] -> Embedding [batch, 38, 768]
        x = self.shared_embedding(encoder_input_tokens.astype("int32"))
        x = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic
        )
        x = x.astype(cfg.dtype)

        # Embedding [batch, 38, 768] -> T5 + ViT [batch, 88, 768]
        for lyr in range(cfg.num_encoder_layers):
            # [batch, length, emb_dim] -> [batch, length, emb_dim]
            x = EncoderLayer(
                config=cfg, relative_embedding=rel_emb, name=f"layers_{lyr}"
            )(x, encoder_mask, deterministic)

        x = layers.LayerNorm(dtype=cfg.dtype, name="encoder_norm")(x)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)
        # Add patch_vit_embs to x
        # [batch, length, emb_dim] -> [batch, length + patch_vit_embs.shape[1], emb_dim]
        # Patch ViT Embedding: [batch, 50, 768]
        # x: [batch, 38, 768] -> concatenate(ViT, T5 Encoder): [batch, 88, 768]
        return jnp.concatenate([x, patch_vit_embs], axis=1)


class MergeT5(T5):
    """A T5 model with text and image inputs.
    Ref:src/models/t5x/t5x/examples/t5/network.py line 278"""

    def setup(self):
        cfg = self.config
        self.shared_embedding = layers.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.emb_dim,
            dtype=cfg.dtype,
            attend_dtype=jnp.float32,  # for logit training stability
            embedding_init=nn.initializers.normal(stddev=1.0),
            one_hot=True,
            name="token_embedder",
        )

        self.encoder = Encoder(config=cfg, shared_embedding=self.shared_embedding)
        self.decoder = Decoder(config=cfg, shared_embedding=self.shared_embedding)

    def encode(
        self,
        encoder_input_tokens,
        patch_vit_embs,
        encoder_segment_ids=None,
        enable_dropout=True,
        **unused_kwargs,
    ):
        """Applies Transformer encoder-branch on the inputs."""
        cfg = self.config
        assert encoder_input_tokens.ndim == 2, (
            f"Expected `encoder_input_tokens` to be of shape (batch, len). "
            f"Got {encoder_input_tokens.shape}"
        )
        
        # Make padding attention mask.
        encoder_mask = layers.make_attention_mask(
            encoder_input_tokens > 0, encoder_input_tokens > 0, dtype=cfg.dtype)
        
        # Add segmentation block-diagonal attention mask if using segmented data.
        if encoder_segment_ids is not None:
            encoder_mask = layers.combine_masks(
                encoder_mask,
                layers.make_attention_mask(
                    encoder_segment_ids, encoder_segment_ids, jnp.equal, dtype=cfg.dtype
                ),
            )

        return self.encoder(
            encoder_input_tokens,
            patch_vit_embs,
            encoder_mask,
            deterministic=not enable_dropout,
        )

    def __call__(
        self,
        encoder_input_tokens,
        decoder_input_tokens,
        decoder_target_tokens,
        patch_vit_embs,
        encoder_segment_ids=None,
        decoder_segment_ids=None,
        encoder_positions=None,
        decoder_positions=None,
        *,
        enable_dropout: bool = True,
        decode: bool = False,
    ):
        """Applies Transformer model on the inputs.

        This method requires both decoder_target_tokens and decoder_input_tokens,
        which is a shifted version of the former. For a packed dataset, it usually
        has additional processing applied. For example, the first element of each
        sequence has id 0 instead of the shifted EOS id from the previous sequence.

        Args:
        encoder_input_tokens: input data to the encoder.
        decoder_input_tokens: input token to the decoder.
        decoder_target_tokens: target token to the decoder.
        patch_vit_embs: patch embeddings from the vision transformer.
        encoder_segment_ids: encoder segmentation info for packed examples.
        decoder_segment_ids: decoder segmentation info for packed examples.
        encoder_positions: encoder subsequence positions for packed examples.
        decoder_positions: decoder subsequence positions for packed examples.
        enable_dropout: Ensables dropout if set to True.
        decode: Whether to prepare and use an autoregressive cache.

        Returns:
        logits array from full transformer.
        """
        encoded = self.encode(
            encoder_input_tokens,
            patch_vit_embs,
            encoder_segment_ids=encoder_segment_ids,
            enable_dropout=enable_dropout,
        )

        return self.decode(
            encoded,
            jnp.concatenate(
                [
                    encoder_input_tokens,
                    jnp.ones((encoder_input_tokens.shape[0], patch_vit_embs.shape[1])),
                ],
                axis=1,
            ),  # only used for masks
            decoder_input_tokens,
            decoder_target_tokens,
            encoder_segment_ids=encoder_segment_ids,
            decoder_segment_ids=decoder_segment_ids,
            decoder_positions=decoder_positions,
            enable_dropout=enable_dropout,
            decode=decode,
        )


class PaLI(nn.Module):
    config: dict
    vit_pretrained_path: str = None
    t5_pretrained_path: str = None

    def setup(self):
        # Vision transformer initialization
        vit_cfg, t5_cfg = unfreeze(self.config["vit"]), unfreeze(self.config["t5"])
        vit_model_config = models_vit_config.MODEL_CONFIGS[vit_cfg["model_name"]]
        vit_model_config.classifier = vit_cfg["classifier"]
        # Model name will be VisionTransformer_0
        self.VisionTransformer_0 = ViT(None, **vit_model_config)

        # T5 model initialization
        gin.parse_config_file(t5_cfg["gin_file"])
        config = gin.get_bindings("t5x.examples.t5.network.T5Config")
        config["dropout_rate"] = t5_cfg["dropout_rate"]
        # Model name will be MergeT5_0
        self.MergeT5_0 = MergeT5(T5Config(**config))

    def __call__(
        self,
        images,
        encoder_input_tokens,
        decoder_input_tokens,
        decoder_target_tokens,
        *,
        vit_train=False,
        enable_dropout=True,
        decode=False,
    ):
        patch_embs = self.VisionTransformer_0(images, train=vit_train)

        outputs = self.MergeT5_0(
            encoder_input_tokens,
            decoder_input_tokens,
            decoder_target_tokens,
            patch_embs,
            enable_dropout=enable_dropout,
            decode=decode,
        )
        return {"logits": outputs, "patch_embs": patch_embs}

    def encode(self, *args, **kwargs):
        return self.MergeT5_0.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.MergeT5_0.decode(*args, **kwargs)


def get_vit_pretrained(config: dict):
    """Loads pretrained ViT model

    Args:
        config (dict): config file load from yaml

    Returns:
        dict: pretrained ViT model
    """
    # Vision transformer pretrained weights
    hpparams = config["hyperparams"]
    vit_cfg = unfreeze(config["vit"])
    vit_model_config = models_vit_config.MODEL_CONFIGS[vit_cfg["model_name"]]
    vit_model_config.classifier = vit_cfg["classifier"]
    vit_model = ViT(None, **vit_model_config)
    vit_variables = jax.jit(
        lambda: vit_model.init(
            jax.random.PRNGKey(0),
            # Discard the "num_local_devices" dimension of the batch for initialization.
            jnp.ones(
                shape=(hpparams["train_batch_size"],)
                + tuple(hpparams["image_size"])
                + (3,)
            ),
            train=False,
        ),
        backend="cpu",
    )()
    # Because the head is included in the pretrained model, we need to add a same head in the model and remove later
    vit_variables = vit_variables.unfreeze()
    vit_variables["params"].update(
        {"head": {"kernel": jnp.ones((10, 768)), "bias": jnp.ones((10,))}}
    )
    vit_pretrained_params = checkpoint.load_pretrained(
        pretrained_path=vit_cfg["pretrained_path"],
        init_params=vit_variables["params"],
        model_config=vit_model_config,
    )
    # Remove the head
    vit_pretrained_params = vit_pretrained_params.unfreeze()
    vit_pretrained_params.pop("head")
    # Add the pretrained weights to the model
    vit_variables["params"] = vit_pretrained_params
    return vit_variables


def get_t5x_pretrained(config: dict):
    """Load pretrained T5x model

    Args:
        config (dict): config file load from yaml

    Returns:
        dict: pretrained T5 model
    """
    hpparams = config["hyperparams"]
    t5_cfg = unfreeze(config["t5"])
    # T5 config by gin
    gin.parse_config_file(t5_cfg["gin_file"])
    config = gin.get_bindings("t5x.examples.t5.network.T5Config")
    config["dropout_rate"] = t5_cfg["dropout_rate"]
    config = T5Config(**config)
    t5_model = T5(config)
    rng = PRNGKey(0)

    # Init variables
    dummy_inputs = {
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
    variables = t5_model.init(rng, **dummy_inputs)

    # Load pretrained weights
    nest_asyncio.apply()
    t5_ckpt = load_t5x_checkpoint(t5_cfg["pretrained_path"])

    variables = variables.unfreeze()
    variables["params"] = t5_ckpt["target"]
    variables = freeze(variables)

    return variables
