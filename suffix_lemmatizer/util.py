# -*- coding: utf8 -*-
""" 
utility functions
"""
from collections import defaultdict
import os
import bz2
import logging


logger = logging.getLogger(__name__)

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


def train_channel_model(suf_sub_func, min_count=1):
    """
    Learn P(lemma|word) base suffix substitution frequencies
    """
    fnm = os.path.join(get_data_dir(), 'corpus.bz2')
    logger.debug('Training channel model ' + fnm)
    M = defaultdict(lambda: defaultdict(float))
    with bz2.BZ2File(fnm) as inf:
        for ln in inf:
            lem, word, n = ln.decode('utf8').rstrip().split('\t')
            if int(n) < min_count:
                continue
            l_suf, w_suf = suf_sub_func(lem, word)
            M[l_suf][w_suf] += int(n)
    for l_suf, w_suf_dict in M.iteritems():
        l_suf_count = sum(w_suf_dict.values())
        for w_suf in w_suf_dict.iterkeys(): 
            M[l_suf][w_suf] /= float(l_suf_count)
    return M


def train_language_model(ngram=1, smoothing=None):
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
    fnm = os.path.join(get_data_dir(), 'corpus.bz2')
    logger.debug('Training language model ' + fnm)
    with bz2.BZ2File(fnm) as inf:
        for ln in inf:
            lem, _, n = ln.decode('utf8').rstrip().split('\t')
            M[lem] += int(n)
    
    if smoothing is None:
        return train_with_no_smoothing(M)
    elif smoothing == 'add-one':
        return train_with_add_one_smoothing(M)
    else:
        raise RuntimeError("Invalid smoothing method " + smoothing)


def get_data_dir():
    return os.path.join(
               os.path.split(os.path.abspath(__file__))[0],
               'data')


def load_dictionary():
    """
    Loads the dictionary from the bz2-compressed file
    """
    fnm = os.path.join(get_data_dir(), 'dict.bz2')
    logger.debug('Loading dictionary ' + fnm)
    D = defaultdict(list)
    with bz2.BZ2File(fnm) as inf:
        for ln in inf:
            infl, lem = ln.decode('utf8').rstrip().split("\t")
            D[infl].append(lem)
    return D
