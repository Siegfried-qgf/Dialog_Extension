import json
import sys
sys.path.append("./../..")
from utils.io_utils import load_json,save_json
path_woz="./../MultiWOZ_2.0/processed/"
path_qa="./../CQA/"
path_cc="./../chit_chat/FusedChat/"
path_recommend="./../CRS/ReDial/"

def cat_data(data1,data2,data3,data4):
    data1=load_json(data1)
    data2=load_json(data2)
    data3=load_json(data3)
    data4=load_json(data4)
    print("TOD :" + str(len(data1)))
    print("QA :" + str(len(data2)))
    print("CC :" + str(len(data3)))
    print("CRS :" + str(len(data4)))
    data1.update(data2)
    data1.update(data3)
    data1.update(data4)
    print(len(data1))
    save_json(data1,"./../multi_data/test_MUL.json")

cat_data(path_woz+'test_TOD.json',path_qa+'test_QA.json',path_cc+"test_CC.json",path_recommend+"test_CRS.json")

'''
def cat_data2(data1,data2):
    data1 = load_json(data1)
    data2 = load_json(data2)
    data1.update(data2)
    save_json(data1, "./../MultiWOZ_2.0/processed/dev_cat_2.json")

cat_data2(path_woz+'dev_data.json',path_cc+"cc_dev.json")
'''
'''
def cat_data3(data1,data2,data3):
    data1=load_json(data1)
    data2=load_json(data2)
    data3=load_json(data3)
    print("TOD :" + str(len(data1)))
    print("CC :" + str(len(data2)))
    print("CRS :" + str(len(data3)))
    data1.update(data2)
    data1.update(data3)
    print(len(data1))
    save_json(data1,"./../multi_data3/train_MUL.json")
'''
#cat_data3(path_woz+'train_TOD.json',path_cc+'train_CC.json',path_recommend+"train_CRS.json")
