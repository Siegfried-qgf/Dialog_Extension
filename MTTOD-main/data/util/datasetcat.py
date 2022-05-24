import json
import sys
sys.path.append("./../..")
from utils.io_utils import load_json,save_json
path_woz="./../MultiWOZ_2.0/processed/"
path_qa="./../CQA/"
path_cc= "../CC/FusedChat/"
path_recommend="./../CRS/ReDial/"
path_squad="./../CQA/Squad/"
path_ubuntu="./../CC/Ubuntu/"
def cat_data(data1,data2,data3,data4,data5,data6):
    data1=load_json(data1)
    data2=load_json(data2)
    data3=load_json(data3)
    data4=load_json(data4)
    data5=load_json(data5)
    data6 = load_json(data6)
    print("TOD :" + str(len(data1)))
    print("QA :" + str(len(data2)))
    print("FusedChat :" + str(len(data3)))
    print("CRS :" + str(len(data4)))
    print("Ubuntu :" + str(len(data5)))
    print("Squad :" + str(len(data6)))
    data1.update(data2)
    data1.update(data3)
    data1.update(data4)
    data1.update(data5)
    data1.update(data6)
    print(len(data1))
    save_json(data1,"./../multi_data/train_MUL.json")

#cat_data(path_woz+'train_TOD.json',path_qa+'train_QA.json',path_cc+"train_CC_FU.json",path_recommend+"train_CRS.json",path_ubuntu+"train_CC_UB.json",path_squad+"train_QA_S.json")

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
def cat_datacc(data1,data2):
    data1=load_json(data1)
    data2=load_json(data2)
    print("FusedChat :" + str(len(data1)))
    print("Ubuntu :" + str(len(data2)))
    data1.update(data2)
    print(len(data1))
    save_json(data1,"./../CC/CC_combined/train_CC.json")

cat_datacc(path_cc+'train_CC_FU.json',path_ubuntu+"train_CC_UB.json")