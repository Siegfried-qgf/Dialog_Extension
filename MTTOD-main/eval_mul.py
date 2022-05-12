'''
written by qgf 2022/5/6
'''
import os
import spacy
import math
import tqdm
import argparse
import logging
from types import SimpleNamespace
from collections import Counter, OrderedDict
from nltk.util import ngrams
from config import CONFIGURATION_FILE_NAME
from reader import MultiWOZReader
from utils import definitions
import json
from utils.io_utils import get_or_create_logger, load_json
from utils.clean_dataset import clean_slot_values
from collections import Counter
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
import pickle as pkl
'''
     "rec_61": [
        {
            "turn_num": 0,
            "turn_domain": [
                "[recommend]"
            ],
            "user": "<bos_user> hello ! have you seen either of these ? avengers: infinity war (2018) or thor_(comics) thor: ragnarok (2017) , omg yes ! i really like them do you have something else for me a superhero movie would be great ! maybe an older superman_(1978_film) movie <eos_user>",
            "usdx": "<bos_user> hello! have you seen either of these? @205163 or @108934 @169419, omg yes! i really like them do you have something else for me a superhero movie would be great! maybe an older @108195 movie <eos_user>",
            "resp": "<bos_resp> have you seen batman_(1989_film_series) <eos_resp>",
            "redx": "<bos_resp> have you seen batman_(1989_film_series) <eos_resp>",
            "match": 47016,
            "bspn": "<bos_belief> [recommend] [value_entity] thor_(comics) superhero_movie superman_(1978_film) <eos_belief>",
            "aspn": "<bos_act> [recommend] [recommend_act] recommend <eos_act>",
            "dbpn": "<bos_db> [db_recommend] spider-man_(2002_film) x-men_(film) ant-man_(film) batman_(1989_film_series) thor_(comics) i_am_thor superman_(1978_film) deadpool_(film) iron_man_(2008_film) the_incredible_hulk_(film) <eos_db>",
            "user_span": {},
            "resp_span": {},
            "bspn_gen": "<bos_belief> [recommend] [value_entity] thor_(comics) superhero_movie superman_(1978_film) <eos_belief>",
            "span": [],
            "bspn_gen_with_span": "<bos_belief> [recommend] [value_entity] thor_(comics) superhero_movie superman_(1978_film) <eos_belief>",
            "dbpn_gen": "<bos_db> [db_recommend] Deadpool_(film) Superman_(1978_film) Iron_Man_(2008_film) The_Incredible_Hulk_(film) The_Avengers_(2012_film) Thor_(comics) X-Men_(film) Spider-Man_(2002_film) Ant-Man_(film) Watchmen_(film) <eos_db>",
            "aspn_gen": "<bos_act> [recommend] [recommend_act] recommend <eos_act>",
            "resp_gen": "<bos_resp> i have n't seen either of those . i would recommend batman_(1989_film_series) and thor: ragnarok (2017) . i have n't seen either of those . <eos_resp>"
        },
'''
action=["[attraction]" ,"[hotel]" ,"[restaurant]" , "[taxi]" , "[train]" , "[hospital]" ,"[police]","[general]"]
def accuracycc(data):
    count=0
    recall=0
    for dial_id, dial in tqdm.tqdm(data.items()):
        for turn in dial:
            count = count + 1
            if "[chit]" in turn["bspn_gen"]:
                recall=recall+1
            else:
                print(dial_id)
    print("匹配到的 "+str(recall))
    print("总数 "+str(count))
    return recall/count

def accuracyqa(data):
    count = 0
    recall = 0
    for dial_id, dial in tqdm.tqdm(data.items()):
        for turn in dial:
            count = count + 1
            if "[answer]" in turn["bspn_gen"]:
                recall = recall + 1
            else:
                print(dial_id)
    print("匹配到的 " + str(recall))
    print("总数 " + str(count))
    return recall / count

def accuracytod(data):
    count = 0
    recall = 0
    for dial_id, dial in tqdm.tqdm(data.items()):
        for turn in dial:
            count = count + 1
            recall = recall + 1
            if "[answer]" in turn["bspn_gen"]:
                recall = recall-1
            elif "[chit]" in turn["bspn_gen"]:
                recall = recall - 1
            elif "[recommend]" in turn["bspn_gen"]:
                recall = recall - 1
            elif turn["bspn_gen"]=="<bos_belief> <eos_belief>":
                print(dial_id)
                recall = recall - 1
    print("匹配到的 " + str(recall))
    print("总数 " + str(count))
    return recall / count

def accuracycrs(data):
    count = 0
    recall = 0
    for dial_id, dial in tqdm.tqdm(data.items()):
        for turn in dial:
            count = count + 1
            recall = recall + 1
            if "[answer]" in turn["bspn_gen"]:
                recall = recall-1
            elif "[chit]" in turn["bspn_gen"]:
                recall = recall - 1
            elif turn["bspn_gen"]=="<bos_belief> <eos_belief>":
                if "[recommend]" not in turn["aspn_gen"]:
                    recall=recall-1
                    print(dial_id)
    print("匹配到的 " + str(recall))
    print("总数 " + str(count))
    return recall / count

def accuracymul(data,type1,type2):
    count = 0
    recall = 0
    neg=0
    for dial_id, dial in tqdm.tqdm(data.items()):
        count=count+1
        for turn in dial:
            if type2=="CC":
                if turn["turn_domain"][0]=="[chit]":
                    if "[chit_act]" in turn["aspn_gen"]:
                        #print(turn["aspn_gen"]+dial_id)
                        recall+=1
                        break
                    else:
                        print(turn["aspn_gen"]+dial_id)
                        neg+=1
                        break
            if type2=="TOD":
                if turn["turn_domain"][0] == "[attraction]" or turn["turn_domain"][0] =="[hotel]" or turn["turn_domain"][0] =="[restaurant]" or turn["turn_domain"][0] =="[taxi]" or turn["turn_domain"][0] =="[train]" or turn["turn_domain"][0] =="[hospital]" or turn["turn_domain"][0] =="[police]":
                    for act in action:
                        if act in turn["aspn_gen"]:
                            recall+=1
                            print(turn["aspn_gen"]+" " +dial_id)
                    break
            if type2=="QA":
                if turn["turn_domain"][0] == "[answer]":
                    if "[answer]" in turn["aspn_gen"]:
                        print(turn["aspn_gen"]+" "+dial_id)
                        recall+=1
                        break
                    else:
                        print(turn["aspn_gen"]+" "+dial_id)
                        neg+=1
                        break
            if type2=="CRS":
                if turn["turn_domain"][0] == "[recommend]":
                    if "[recommend]" in turn["aspn_gen"]:
                        #print(turn["aspn_gen"]+" "+dial_id)
                        recall+=1
                        break
                    else:
                        print(turn["aspn_gen"]+" "+dial_id)
                        neg+=1
                        break
    print("匹配到的 " + str(recall))
    print("总数 " + str(count))
    print("neg"+str(neg))
    return recall / count
cc=load_json("./MUL_5.6/ckpt-epoch5/CC")
qa=load_json("./MUL_5.6/ckpt-epoch5/QA")
tod=load_json("./MUL_5.6/ckpt-epoch5/TOD")
crs=load_json("./MUL_5.6/ckpt-epoch5/CRS")
tod_cc=load_json("./MUL_5.6/ckpt-epoch5/tod_cc")
cc_tod=load_json("./MUL_5.6/ckpt-epoch5/cc_tod")
qa_tod=load_json("./MUL_5.6/ckpt-epoch5/qa_tod")
tod_qa=load_json("./MUL_5.6/ckpt-epoch5/tod_qa")
crs_tod=load_json("./MUL_5.6/ckpt-epoch5/crs_tod")
tod_crs=load_json("./MUL_5.6/ckpt-epoch5/tod_crs")
cc_qa=load_json("./MUL_5.6/ckpt-epoch5/cc_qa")
qa_cc=load_json("./MUL_5.6/ckpt-epoch5/qa_cc")
cc_crs=load_json("./MUL_5.6/ckpt-epoch5/cc_crs")
crs_cc=load_json("./MUL_5.6/ckpt-epoch5/crs_cc")
qa_crs=load_json("./MUL_5.6/ckpt-epoch5/qa_crs")
crs_qa=load_json("./MUL_5.6/ckpt-epoch5/crs_qa")
accuracymul(crs_qa,"CRS","QA")