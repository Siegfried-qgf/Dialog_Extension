'''
written by qgf 2022/3/28
'''
import os
import spacy
import math
import argparse
import logging
from types import SimpleNamespace
from collections import Counter, OrderedDict
from nltk.util import ngrams
from config import CONFIGURATION_FILE_NAME
from reader import MultiWOZReader
from utils import definitions
from utils.io_utils import get_or_create_logger, load_json
from utils.clean_dataset import clean_slot_values
from collections import Counter
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
logger = get_or_create_logger(__name__)
nlp = spacy.load('en_core_web_sm')
class CC_evaluator(object):
    def __init__(self, reader):
        self.reader = reader
    '''
    'sng01270_cc': [{'turn_num': 0, 'turn_domain': ['[chit]'], 'pointer': '',
                     'user': "<bos_user> people don't usually drive in singapore. <eos_user>",
                     'resp': '<bos_resp> they have convenient and complete public transport and taxis that can reach anywhere on the island. <eos_resp>',
                     'redx': '<bos_resp> they have convenient and complete public transport and taxis that can reach anywhere on the island. <eos_resp>',
                     'bspn': '<bos_belief> <eos_belief>', 'aspn': '<bos_act> [chit] [chit_act] chit <eos_act>',
                     'dbpn': '<bos_db> [db_null] <eos_db>', 'user_span': {}, 'resp_span': {},
                     'bspn_gen': '<bos_belief> <eos_belief>', 'span': [],
                     'bspn_gen_with_span': '<bos_belief> <eos_belief>', 'dbpn_gen': '<bos_db> [db_null] <eos_db>',
                     'aspn_gen': '<bos_act> [chit] [chit_act] <eos_act>',
                     'resp_gen': '<bos_resp> it is convenient to commute by public transport like train these days. <eos_resp>'},
                    {'turn_num': 1, 'turn_domain': ['[chit]'], 'pointer': '',
                     'user': '<bos_user> they have convenient and complete public transport and taxis that can reach anywhere on the island. <eos_user>',
                     'resp': '<bos_resp> i am an adult but i do not know how to drive. <eos_resp>',
                     'redx': '<bos_resp> i am an adult but i do not know how to drive. <eos_resp>',
                     'bspn': '<bos_belief> <eos_belief>', 'aspn': '<bos_act> [chit] [chit_act] chit <eos_act>',
                     'dbpn': '<bos_db> [db_null] <eos_db>', 'user_span': {}, 'resp_span': {},
                     'bspn_gen': '<bos_belief> <eos_belief>', 'span': [],
                     'bspn_gen_with_span': '<bos_belief> <eos_belief>', 'dbpn_gen': '<bos_db> [db_null] <eos_db>',
                     'aspn_gen': '<bos_act> [chit] [chit_act] <eos_act>',
                     'resp_gen': '<bos_resp> singapore is a well-developed country. <eos_resp>'}]
    '''
    def dist(self,data):
        unigrams_all, bigrams_all = Counter(), Counter()
        resp = []
        for dial_id,dial in data.items():
            for turn in dial:
                turn_resp_=nlp(turn['resp_gen'][11:-11])
                turn_resp=[]
                for token in turn_resp_:
                    turn_resp.append(str(token))
                resp.append(turn_resp)
        for seq in resp:
            unigrams = Counter(seq)
            bigrams = Counter(zip(seq, seq[1:]))
            unigrams_all.update(unigrams)
            bigrams_all.update(bigrams)
        #print(unigrams_all)
        #print(bigrams_all)
        inter_dist1 = (len(unigrams_all) + 1e-12) / (sum(unigrams_all.values()) + 1e-5)
        inter_dist2 = (len(bigrams_all) + 1e-12) / (sum(bigrams_all.values()) + 1e-5)
        return inter_dist1,inter_dist2


    def bleu(self, data):
        """ Calculate bleu 1/2. """
        resp=[]
        gold_resp=[]
        for dial_id,dial in data.items():
            for turn in dial:
                turn_resp=turn['resp_gen'][11:-11]
                resp.append(turn_resp)
                gold_turn_resp=turn['resp'][11:-11]
                gold_resp.append(gold_turn_resp)

        bleu_1 = []
        bleu_2 = []
        for hyp, ref in zip(resp, gold_resp):
            try:
                score = bleu_score.sentence_bleu(
                    [ref], hyp,
                    smoothing_function=SmoothingFunction().method7,
                    weights=[1, 0, 0, 0])
            except:
                score = 0
            bleu_1.append(score)
            try:
                score = bleu_score.sentence_bleu(
                    [ref], hyp,
                    smoothing_function=SmoothingFunction().method7,
                    weights=[0.5, 0.5, 0, 0])
            except:
                score = 0
            bleu_2.append(score)
        bleu_1 = np.average(bleu_1)
        bleu_2 = np.average(bleu_2)
        return bleu_1, bleu_2


