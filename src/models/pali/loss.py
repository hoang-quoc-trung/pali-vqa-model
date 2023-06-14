import jax
import optax

def cross_entropy_loss(logits, targets, vocab_size):
    targets_1hot = jax.nn.one_hot(targets, num_classes=vocab_size, axis=-1)
    return optax.softmax_cross_entropy(logits=logits, labels=targets_1hot).mean()