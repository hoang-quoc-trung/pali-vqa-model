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

def bleu_score(logits, targets):
    mean_blue_score = 0
    predicted_tokens = argmax_logits(logits)
    def single_bleu_score(l, t):
        ref = vocab_encoder_decoder.decode(t)
        hypo = vocab_encoder_decoder.decode(l)
        print(f"Y: {ref} | Yhat: {hypo}")
        try:
            result = google_bleu.compute(predictions=[hypo], references=[[ref]])
            result = result['google_bleu']
        except:
            result=0.0
        return result
    for l, t in zip(predicted_tokens, targets):
        mean_blue_score += single_bleu_score(l, t)
    return mean_blue_score / len(logits)

def cider_score(logits, targets):
    # shape(logits): (batch size, 18, 32128) - shape(target): (batch size, 18)
    final_scores = {}
    # shape(predicted_tokens): (3,18)
    predicted_tokens = argmax_logits(logits)
    def convert_to_text(logit, target):
        ref = vocab_encoder_decoder.decode(target)
        hypo = vocab_encoder_decoder.decode(logit)
        # print(f"Y: {ref} | Yhat: {hypo}")
        return ref, hypo
    y_arr = []
    y_hat_arr = []
    for logit, target in zip(predicted_tokens, targets):
        y, y_hat = convert_to_text(logit, target)
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

def combine_metrics(logits, targets):
    # shape(logits): (batch size, 18, 32128) - shape(target): (batch size, 18)
    cider_scores = {}
    mean_blue_score = 0
    predicted_tokens = argmax_logits(logits)
    # shape(predicted_tokens): (3,18)
    predicted_tokens = argmax_logits(logits)
    def convert_to_text(logit, target):
        ref = vocab_encoder_decoder.decode(target)
        hypo = vocab_encoder_decoder.decode(logit)
        # print(f"Y: {ref} | Yhat: {hypo}")
        try:
            bleu_result = google_bleu.compute(predictions=[hypo], references=[[ref]])
            bleu_result = result['google_bleu']
        except:
            bleu_result = 0.0
        return ref, hypo, bleu_result
    y_arr = []
    y_hat_arr = []
    for logit, target in zip(predicted_tokens, targets):
        y, y_hat, bleu = convert_to_text(logit, target)
        y_arr.append(y)
        y_hat_arr.append(y_hat)
        mean_blue_score = mean_blue_score + bleu
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