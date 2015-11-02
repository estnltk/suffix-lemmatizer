# -*- coding: utf8 -*-
""" Utility functions """  

from collections import defaultdict
import codecs
import pandas as pd
from functools import partial
import os
import re


exceptions = dict([
         (u'selle', u'see'),
         (u'seda', u'see'),
         (u'pärast', u'pärast'),
         (u'on', u'olema'),
         (u'või', u'või'),
         (u'&sect;', u'&sect'),
         (u'ta', u'tema'),
         (u'eest', u'eest'),
         (u'kõige', u'kõige'),
         (u'seal', u'seal'),
         (u'kokku', u'kokku'),
         (u'poole', u'poole'),
         (u'alati', u'alati'),
         (u'puhul', u'puhul'),
         (u'korras', u'kord'),
         (u'läks', u'minema'),
         (u'umbes', u'umbes'),
         (u'vastavalt', u'vastavalt'),
         (u'ajal', u'ajal'),
         (u'võiks', u'võima'),
         (u'neil', u'tema'),
         (u'neist', u'see'),
         (u'kehtestatud', u'kehtestatud'),
         (u'läinud', u'minema'),
         (u'meile', u'mina'),
         (u'siia', u'siia'),
         (u'poleks', u'olema'),
         (u'kõrvale', u'kõrvale'),
         (u'kohaselt', u'kohaselt'),
         (u'kohal', u'kohal'),
         (u'antud', u'andma'),
         (u'ringi', u'ringi'),
         (u'saanud', u'saanud'),
         (u'muutus', u'muutuma'),
         (u'lubatud', u'lubatud'),
         (u'teed', u'tee'),
         (u'pidevalt', u'pidevalt'),
         (u'esile', u'esile'),
         (u'kohta', u'kohta'),
         (u'ainult', u'ainult'),
         (u'sest', u'sest'),
         (u'meie', u'mina'),
         (u'nende', u'see'),
         (u'talle', u'tema'),
         (u'need', u'see'),
         (u'rt', u'rt'),
         (u'tal', u'tema'),
         (u'ikka', u'ikka'),
         (u'korral', u'korral'),
         (u'euroopa', u'euroopa'),
         (u'sellele', u'see'),
         (u'sellega', u'see'),
         (u'tavaliselt', u'tavaliselt'),
         (u'lubatud', u'lubama'),
         (u'käes', u'käsi'),
         (u'teid', u'sina'),
         (u'koondumise', u'koondumine'),
         (u'oluliselt', u'oluliselt'),
         (u'nimelt', u'nimelt'),
         (u'alates', u'alates'),
         (u'pähe', u'pea'),
         (u'püsti', u'püsti'),
         (u'lähedal', u'lähedal'),
         (u'inglise', u'inglise'),
         (u'ära', u'ärama'),
         (u'nimel', u'nimel'),
         (u'eestisse', u'eesti'),
         (u'sõlmitud', u'sõlmitud'),
    ])


def inverse_channel_model(chnl_model):
    D = defaultdict(set)
    for l_suf, w_suf_dict in chnl_model.iteritems():
        for w_suf in w_suf_dict.iterkeys():
            D[w_suf].add(l_suf)
    return D


def train_channel_model(train_fl, suf_sub_func, min_count=1):
    """ Learn P(lemma|word) base suffinx substitution frequencies. """
    M = defaultdict(lambda: defaultdict(float))
    with codecs.open(train_fl, encoding='utf8') as inf:
        for ln in inf:
            ln = ln.lower().rstrip()
            lem, word, n = ln.split('\t')
            if int(n) < min_count:
                continue
            l_suf, w_suf = suf_sub_func(lem, word)
            M[l_suf][w_suf] += int(n)
    for l_suf, w_suf_dict in M.iteritems(): # optional: try to prune lemmas/rules? with low confidence
        l_suf_count = sum(w_suf_dict.values())
        for w_suf in w_suf_dict.iterkeys(): 
            M[l_suf][w_suf] /= float(l_suf_count)
    return M

def dict_lookup_exact(dic, word, **kwargs):
    return '', word, dic.get(word)

def dict_lookup_suffix(dic, word, **kwargs):
    """ Returns (prefix, suffix, candidate-lemmas) using longest suffix match """
    min_suf_len = kwargs.get('min_suf_len', 5)
    for i in xrange(max(len(word) - min_suf_len, 1)):
        if word[i:] in dic:
            return word[:i], word[i:], dic[word[i:]]
    else:
        return '', word, []


def dict_lookup_pref_suf(dic, word, **kwargs):
    min_suf_len = kwargs.get('min_suf_len', 5)
    for i in xrange(max(len(word) - min_suf_len, 1)):
        pref, suff = word[:i], word[i:]
        if len(pref) > 2:
            if pref in dic and suff in dic:
                return word[:i], word[i:], dic[word[i:]]
        else:
            if suff in dic:
                return word[:i], word[i:], dic[word[i:]]
    else:
        return '', word, []


def train_language_model(train_fl, ngram=1, smoothing=None):
    """
    Train P(L) part of the model: P(W|L)P(L)
    """

    def train_with_no_smoothing(lemma2count_dict):
        total_lem_count = float(sum(lemma2count_dict.itervalues()))
        for lem in lemma2count_dict:
            lemma2count_dict[lem] /= total_lem_count
        return lemma2count_dict
        
    def train_with_add_one_smoothing(lemma2count_dict):
        total_lem_count  = sum(lemma2count_dict.itervalues())
        unique_lem_count = len(lemma2count_dict)
        for lem in lemma2count_dict:
            lemma2count_dict[lem] = (lemma2count_dict[lem] + 1.0) /\
                                    (total_lem_count + unique_lem_count)
        res = defaultdict(lambda: 1.0 / (total_lem_count + unique_lem_count))
        res.update(lemma2count_dict)
        return res

    M = defaultdict(int)
    with codecs.open(train_fl, encoding='utf8') as inf:
        for ln in inf:
            lem, _, n = ln.rstrip().split('\t')
            M[lem] += int(n)
    
    if smoothing is None:
        return train_with_no_smoothing(M)
    elif smoothing == 'add-one':
        return train_with_add_one_smoothing(M)
    else:
        raise RuntimeError("Invalid smoothing method " + smoothing)


def load_data_lwc(fnm, toLower=True, topN=100000000):
    D = defaultdict(set)
    with codecs.open(fnm, encoding='utf8') as inf:
        for i, ln in enumerate(inf):
            if i == topN:
                break
            if toLower:
                ln = ln.lower()
            lem, infl, n = ln.strip().split("\t")
            D[infl].add(lem)
    D = dict((infl, list(lemma_set)) for infl, lemma_set in D.iteritems())
    return D
