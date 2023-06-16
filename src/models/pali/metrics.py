from t5.data import get_default_vocabulary as get_t5_vocab
import jax
import jax.numpy as jnp
import evaluate
from models.pali.cider import Cider

vocab_encoder_decoder = get_t5_vocab()
google_bleu = evaluate.load("google_bleu")
scorers = [(Cider(), "CIDEr")]

@jax.jit
def argmax_logits(logits):
    return jnp.argmax(logits, axis=-1)


def convert_to_text(logit_tokens, target_tokens):
    hypothesis = vocab_encoder_decoder.decode(logit_tokens)
    references = vocab_encoder_decoder.decode(target_tokens)
    # print(f"Y: {references} | Yhat: {hypothesis}")
    return hypothesis, references


def bleu_score(logits, target_tokens):
    mean_blue_score = 0.0
    logit_tokens = argmax_logits(logits)
    for l, t in zip(logit_tokens, target_tokens):
        hypothesis, references  = convert_to_text(l, t)
        try:
            result = google_bleu.compute(predictions=[hypothesis], references=[[references]])
            result = result['google_bleu']
        except:
            result=0.0
        mean_blue_score += result
    return mean_blue_score / len(logits)


def cider_score(logits, target_tokens):
    final_scores = {}
    y_arr = []
    y_hat_arr = []
    # shape(logits): (batch size, 18, 32128) - shape(target): (batch size, 18)
    # shape(logit_tokens): (3,18)
    logit_tokens = argmax_logits(logits)
    for l, t in zip(logit_tokens, target_tokens):
        y_hat, y = convert_to_text(l, t)
        y_arr.append(y)
        y_hat_arr.append(y_hat)
    # shape(y_hat_arr and y_arr): (batch size,)
    # convert the shape of y_arr to (batch size, 1)
    reference = [ [x] for x in y_arr ]
    # convert the shape of y_hat_arr to (batch size, 1)
    hypothesis = [ [x] for x in y_hat_arr ]
    for scorer, method in scorers:
        # input shape of compute_score function is (batch size, 1)
        score, scores = scorer.compute_score(reference, hypothesis)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s    
        else:
            final_scores[method] = score  
    return final_scores 


def combine_metrics(logits, target_tokens):
    cider_scores = {}
    mean_blue_score = 0.0
    y_arr = []
    y_hat_arr = []
    # shape(logits): (batch size, 18, 32128) - shape(target): (batch size, 18)
    # shape(logit_tokens): (3,18)
    logit_tokens = argmax_logits(logits)
    for l, t in zip(logit_tokens, target_tokens):
        y_hat, y = convert_to_text(l, t)
        y_arr.append(y)
        y_hat_arr.append(y_hat)
        # print(f"Y: {y} | Yhat: {y_hat}")
        try:
            bleu_result = google_bleu.compute(predictions=[y_hat], references=[[y]])
            bleu_result = bleu_result['google_bleu']
        except:
            bleu_result = 0.0
        mean_blue_score += bleu_result
    bleu_score = mean_blue_score / len(logits)
    # shape(y_hat_arr and y_arr): (batch size,)
    # convert the shape of y_arr to (batch size, 1)
    reference = [ [x] for x in y_arr ]
    # convert the shape of y_hat_arr to (batch size, 1)
    hypothesis = [ [x] for x in y_hat_arr ]
    for scorer, method in scorers:
        # input shape of compute_score function is (batch size, 1)
        score, scores = scorer.compute_score(reference, hypothesis)
        if type(score) == list:
            for m, s in zip(method, score):
                cider_scores[m] = s    
        else:
            cider_scores[method] = score  
    return float(cider_scores["CIDEr"]), float(bleu_score)