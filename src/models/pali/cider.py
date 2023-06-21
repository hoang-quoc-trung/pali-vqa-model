import copy
import math
from collections import defaultdict
import numpy as np


def precook(s, n=4, out=False):
    """
    Precook a text by counting n-grams.

    Args:
        s (str): The input text.
        n (int): The maximum n-gram size. Defaults to 4.
        out (bool): Whether to output the n-gram counts. Defaults to False.

    Returns:
        dict or None: If `out` is True, returns a dictionary representing the counts of n-grams in the input text.
                      Otherwise, returns None.
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4):
    """
    Precook a list of reference texts by counting n-grams.

    Args:
        refs (list): A list of reference texts.
        n (int): The maximum n-gram size. Defaults to 4.

    Returns:
        list: A list of dictionaries, where each dictionary represents the counts of n-grams in a reference text.
    """
    return [precook(ref, n) for ref in refs]


def cook_test(test, n=4):
    """
    Precook a test text by counting n-grams.

    Args:
        test (str): The test text.
        n (int): The maximum n-gram size. Defaults to 4.

    Returns:
        dict: A dictionary representing the counts of n-grams in the test text.
    """
    return precook(test, n, True)


class CiderScorer(object):
    def copy(self):
        ''' copy the refs.'''
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        ''' singular instance '''
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        self.cook_append(test, refs)
        self.ref_len = None
    
    def cook_append(self, test, refs):
        '''called by constructor and __iadd__ to avoid creating new instances.'''

        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                self.ctest.append(cook_test(test)) ## N.B.: -1
            else:
                self.ctest.append(None) # lens of crefs and ctest have to match
                
    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)
    
    
    def __iadd__(self, other):
        '''add an instance (e.g., from another sentence).'''

        if type(other) is tuple:
            ## avoid creating new CiderScorer instances
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)

        return self
    
    def compute_doc_freq(self):
        '''
        Compute the document frequency for the reference data.

        This function calculates the term frequency for the reference data, which will be used to compute the
        inverse document frequency (idf) later. The term frequency is stored in the object.
        '''
        for refs in self.crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
                self.document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
    
    def compute_cider(self):
        def counts2vec(cnts):
            """
            Maps counts of n-grams to a vector of tf-idf weights.

            This function takes the counts of n-grams and returns a vector `vec`, which is an array of dictionaries.
            Each dictionary stores the mapping of an n-gram to its tf-idf weight. The n-th entry of the array denotes
            the length of n-grams.

            Args:
                cnts: A dictionary containing the counts of n-grams.

            Returns:
                vec: An array of dictionaries, representing the tf-idf weights for each n-gram length.
                norm: An array of floats, representing the norm of each vector.
                length: An integer, representing the length of 1-grams.
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram,term_freq) in cnts.items():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self.document_frequency[ngram]))
                # ngram index
                n = len(ngram)-1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq)*(self.ref_len - df)
                # compute norm for the vector.  the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length
        
        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            """
            Compute the cosine similarity of two vectors.

            Args:
                vec_hyp: An array of dictionaries representing the vector corresponding to the hypothesis.
                vec_ref: An array of dictionaries representing the vector corresponding to the reference.
                norm_hyp: An array of floats representing the norm of the hypothesis vector.
                norm_ref: An array of floats representing the norm of the reference vector.
                length_hyp: An integer containing the length of the hypothesis.
                length_ref: An integer containing the length of the reference.

            Returns:
                An array of scores for each n-gram's cosine similarity.
            """
            delta = float(length_hyp - length_ref)
            # measure consine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram,count) in vec_hyp[n].items():
                    # vrama91 : added clipping
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n]*norm_ref[n])

                assert(not math.isnan(val[n]))
                # vrama91: added a length based gaussian penalty
                val[n] *= np.e**(-(delta**2)/(2*self.sigma**2))
            return val

        # compute log reference length
        self.ref_len = np.log(float(len(self.crefs)))

        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            # compute vector for test captions
            vec, norm, length = counts2vec(test)
            # compute vector for ref captions
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score)
            # divide by number of references
            score_avg /= len(refs)
            # multiply score by 10
            score_avg *= 10.0
            # append score of an image to the score list
            scores.append(score_avg)
        return scores
    
    def compute_cider_score(self, option=None, verbose=0):
        # compute idf
        self.compute_doc_freq()
        # assert to check document frequency
        assert(len(self.ctest) >= max(self.document_frequency.values()))
        # compute cider score
        score = self.compute_cider()
        # print score
        return np.mean(np.array(score)), np.array(score)
    
    
class Cider:
    """
    Main Class to compute the CIDEr metric 
    """
    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma

    def compute_cider_score(self, gts, res):
        """
        Main function to compute CIDEr score.

        Args:
            gts (dict): A dictionary with keys as <image> and values as tokenized reference sentences.
            res (dict): A dictionary with keys as <image> and values as tokenized hypothesis/candidate sentences.

        Returns:
            cider (float): Computed CIDEr score for the corpus.
            scores (list): List of individual CIDEr scores for each image.
        """
        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)
        for id in range(len(res)):
            hypo = res[id]
            ref = gts[id]
            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)
            cider_scorer += (hypo[0], ref)
        score, scores = cider_scorer.compute_cider_score()
        return score, scores

    def method(self):
        return compute_cider_score