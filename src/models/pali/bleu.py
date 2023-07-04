from typing import Dict, List
from nltk.translate import gleu_score
from models.pali.bleu_tokenizer import Tokenizer13a


def compute_bleu(
    predictions: List[str],
    references: List[List[str]],
    tokenizer=Tokenizer13a(),
    min_len: int = 1,
    max_len: int = 4,
) -> Dict[str, float]:
    """
    Compute BLEU score for machine translation evaluation.
    
    Args:
        predictions (List[str]): List of predicted sentences.
        references (List[List[str]]): List of reference sentences. Each reference sentence is a list of strings.
        tokenizer: Tokenizer used to tokenize the sentences. Default is Tokenizer13a().
        min_len (int): Minimum n-gram order to consider for BLEU score calculation. Default is 1.
        max_len (int): Maximum n-gram order to consider for BLEU score calculation. Default is 4.
    
    Returns:
        BLEU score (float)
    
    Note:
        The function assumes that if only one reference is provided, it should still be in the format of a list of lists.
    """
    if isinstance(references[0], str):
        references = [[ref] for ref in references]
    
    references = [[tokenizer(r) for r in ref] for ref in references]
    predictions = [tokenizer(p) for p in predictions]
    return gleu_score.corpus_gleu(
        list_of_references=references, 
        hypotheses=predictions, 
        min_len=min_len, 
        max_len=max_len
    )