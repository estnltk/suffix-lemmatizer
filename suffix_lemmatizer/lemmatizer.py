# -*- coding: utf8 -*-
import re
import random
import logging

from . import util


log = logging.getLogger(__name__)

class SuffixLemmatizer(object):
    """
    Lemmatizer which handles in-dictionary and out-of-dictionary words.
    """
    def __init__(self):
        log.debug('Loading suffix lemmatizer ...')
        self.dic = util.load_dictionary()
        self.chnl_model = util.train_channel_model(suf_sub_func=self.get_suffix_sub)
        self.wsuf2lsuf_dict = util.inverse_channel_model(self.chnl_model)
        self.lng_model = util.train_language_model()
        
    
    def __call__(self, word):
        """
        Lemmatizes a given word
        """
        if word in util.exceptions:
            return util.exceptions[word]
        return self.lemmatize_dict(word) or  self.lemmatize_oov(word) or word
    
    
    def dict_lookup_suffix(self, word, min_suf_len=5):
        """ Returns (prefix, suffix, candidate-lemmas) using longest suffix match """
        for i in xrange(max(len(word) - min_suf_len, 1)):
            if word[i:] in self.dic:
                return word[:i], word[i:], self.dic[word[i:]]
        else:
            return '', word, []
    
    
    def get_suffix_sub(self, lemma, inflection, pref_len=1):
        maxi = min(len(lemma), len(inflection))
        i = 0
        while i < maxi and lemma[i] == inflection[i]:
            i += 1
        suf_lem = lemma[i:]
        suf_inf = inflection[i:]
        for j in xrange(i-1, i-pref_len-1, -1):
            pref = lemma[j] if j >= 0 else '$'
            suf_lem = pref + suf_lem
            suf_inf = pref + suf_inf
        return (suf_lem, suf_inf)
    
    
    def lemmatize_dict(self, word):
        """
        Dictionary-base lemmatization: If multiple candidates found, 
        score candidates base on channel model & language model.
        """
        pref, _, candidates = self.dict_lookup_suffix(word)
        if candidates and len(candidates) > 1:
            best_candidates, max_score = [], -1
            for candidate in candidates:
                l_suf, w_suf = self.get_suffix_sub(candidate, word)
                w1 = self.chnl_model[l_suf][w_suf]
                w2 = max(self.lng_model[candidate], self.lng_model[pref + candidate])
                score = w1 * w2
                if score > max_score:
                    best_candidates = [candidate] 
                    max_score = score
                elif abs(score - max_score) < 0.000001:
                    best_candidates.append(candidate)
            return pref + best_candidates[random.randint(0, len(best_candidates) - 1)]
        elif len(candidates)==1:
            return candidates[0]
        else:
            return None

    
    def lemmatize_oov(self, word):
        """ 
        Handles OOV case: Word not in dictionary, 
        score candidates using channel model & language model.
        """
        candidates = []
        for i in xrange(len(word) + 1):
            w_pref, w_suf = word[:i], word[i:]
            for l_suf in self.wsuf2lsuf_dict[w_suf]:
                lemma = w_pref + l_suf
                if lemma in lng_model:
                    p = (self.lng_model[lemma] ** 0.8) * self.chnl_model[l_suf][w_suf]
                    candidates.append((p, lemma))
        candidates.sort(reverse=1)
        if candidates:
            return candidates[0][1]
        else:
            return None
