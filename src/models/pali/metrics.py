from t5.data import get_default_vocabulary as get_t5_vocab
import jax
import jax.numpy as jnp
from models.pali.cider import Cider
from models.pali.bleu import compute_bleu

vocab_encoder_decoder = get_t5_vocab()
cider = Cider()

@jax.jit
def argmax_logits(logits):
    """Compute the index of the maximum logit value along the last axis.
    
    Args:
        logits (jax.ndarray): Logits tensor.
    
    Returns:
        jax.ndarray: Index of the maximum logit value along the last axis.
    """
    return jnp.argmax(logits, axis=-1)


def convert_to_text(logit_tokens, target_tokens):
    """Convert tokenized logit and target sequences to text.
    
    Args:
        logit_tokens (List[List[int]]): List of logit token sequences.
        target_tokens (List[List[int]]): List of target token sequences.
    
    Returns:
        Tuple[List[str], List[str]]: Tuple containing hypothesis (decoded logit sequences)
        and references (decoded target sequences).
    """
    hypothesis = vocab_encoder_decoder.decode(logit_tokens)
    references = vocab_encoder_decoder.decode(target_tokens)
    print(f"Y: {references} | Yhat: {hypothesis}")
    return hypothesis, references


def bleu_score(logits, target_tokens):
    """Convert tokenized logit and target sequences to text.
    
    Args:
        logit_tokens (List[List[int]]): List of logit token sequences.
        target_tokens (List[List[int]]): List of target token sequences.
    
    Returns:
        Tuple[List[str], List[str]]: Tuple containing hypothesis (decoded logit sequences)
        and references (decoded target sequences).
    """
    mean_blue_score = 0.0
    logit_tokens = argmax_logits(logits)
    for l, t in zip(logit_tokens, target_tokens):
        hypothesis, references  = convert_to_text(l, t)
        try:
            result = compute_bleu(predictions=[hypothesis], references=[[references]])
        except:
            result=0.0
        mean_blue_score += result
    return mean_blue_score / len(logits)


def cider_score(logits, target_tokens):
    """Compute the CIDEr score between predicted logits and target tokens.
    
    Args:
        logits (Array): Predicted logits with shape (batch_size, length, vocab).
        target_tokens (List[List[int]]): List of target token sequences.
    
    Returns:
        dict: Dictionary containing the CIDEr scores.
    """
    final_scores = {}
    y_arr = []
    y_hat_arr = []
    # shape(logits): (batch_size, length, vocab) - shape(target): (batch_size, length)
    # shape(logit_tokens): (batch_size, length)
    logit_tokens = argmax_logits(logits)
    for l, t in zip(logit_tokens, target_tokens):
        y_hat, y = convert_to_text(l, t)
        y_arr.append(y)
        y_hat_arr.append(y_hat)
    # shape(y_hat_arr and y_arr): (batch_size,)
    # convert the shape of y_arr to (batch_size, 1)
    reference = [ [x] for x in y_arr ]
    # convert the shape of y_hat_arr to (batch_size, 1)
    hypothesis = [ [x] for x in y_hat_arr ]
    cider_score, cider_scores = cider.compute_cider_score(reference, hypothesis) 
    return cider_score


def combine_metrics(logits, target_tokens):
    """Combine multiple evaluation metrics including CIDEr and BLEU scores.
    
    Args:
        logits (Array): Predicted logits with shape (batch_size, 18, 32128).
        target_tokens (List[List[int]]): List of target token sequences.
    
    Returns:
        Tuple[float, float]: CIDEr score and BLEU score as a tuple.
    """
    
    mean_blue_score = 0.0
    y_arr = []
    y_hat_arr = []
    # shape(logits): (batch_size, length, vocab) - shape(target): (batch_size, length)
    # shape(logit_tokens): (batch_size, length)
    logit_tokens = argmax_logits(logits)
    for l, t in zip(logit_tokens, target_tokens):
        y_hat, y = convert_to_text(l, t)
        y_arr.append(y)
        y_hat_arr.append(y_hat)
        # shape(y_hat_arr and y_arr): (batch_size,)
        try:
            bleu_result = compute_bleu(predictions=[y_hat], references=[[y]])
        except:
            bleu_result = 0.0
        mean_blue_score += bleu_result
    bleu_score = mean_blue_score / len(logits)
    # convert the shape of y_arr to (batch_size, 1)
    reference = [ [x] for x in y_arr ]
    # convert the shape of y_hat_arr to (batch_size, 1)
    hypothesis = [ [x] for x in y_hat_arr ]
    cider_score, cider_scores = cider.compute_cider_score(reference, hypothesis) 
    return cider_score, bleu_score