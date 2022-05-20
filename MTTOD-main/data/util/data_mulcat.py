import copy
import json
import sys
import tqdm
import random
sys.path.append("./../..")
from utils.io_utils import load_json,save_json
path_woz="./../MultiWOZ_2.0/processed/"
path_qa="./../CQA/"
path_cc= "../CC/FusedChat/"
path_recommend="./../CRS/ReDial/"

test_woz=load_json(path_woz+"test_TOD.json")
test_cc=load_json(path_cc+"test_CC.json")
test_qa=load_json(path_qa+"test_QA.json")
test_crs=load_json(path_recommend+"test_CRS.json")

def tod_cc(datatod,datacc):
    tod2cc = {}
    for key,item in tqdm.tqdm(datatod.items()):
        key=key[:-5]+"_cc"
        count=len(item["log"])
        for i in datacc[key]["log"]:
            i["turn_num"]=i["turn_num"]+count
        for turn in datacc[key]["log"]:
            item["log"].append(turn)
        tod2cc.update({key:item})
    save_json(tod2cc,"./../multi_scene/tod_cc.json")

def cc_tod(datatod,datacc):
    cc2tod = {}
    for key,item in tqdm.tqdm(datacc.items()):
        key=key[:-3]+".json"
        count=len(item["log"])
        for i in datatod[key]["log"]:
            i["turn_num"]=i["turn_num"]+count
        for turn in datatod[key]["log"]:
            item["log"].append(turn)
        cc2tod.update({key:item})
    save_json(cc2tod,"./../multi_scene/cc_tod.json")

def mulcat(data1,data2,num):
    result={}
    for key,item in tqdm.tqdm(data1.items()):
        count=len(item["log"])
        ran=random.sample(list(data2.items()),num)
        for i in range(num):
            temp=copy.deepcopy(item)
            id,dict=ran[i]
            for j in dict["log"]:
                temp2=copy.deepcopy(j)
                temp2["turn_num"]=temp2["turn_num"]+count
                temp["log"].append(temp2)
            result.update({(key+"_"+str(i)):temp})
    save_json(result,"./../multi_scene/crs_qa.json")




mulcat(test_crs,test_qa,2)
#def cat_muldata()