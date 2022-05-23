'''
by qgf 2022/3/19
'''
import json
import sys
import tqdm
sys.path.append("./../../..")
from utils.io_utils import load_json,save_json,load_text
import pandas as pd

train_data=pd.read_csv("./train.csv")
dev_data=pd.read_csv("valid.csv")
test_data=pd.read_csv("test.csv")
train_data=train_data[train_data["Label"]==1]
print(len(train_data))
from transformers import T5Tokenizer
tok=T5Tokenizer.from_pretrained("t5-base")
train_data=train_data.sample(n=100000,random_state=21)

def tran(data,type):
    d={}
    signal=0
    for index, row in tqdm.tqdm(data.iterrows()):
        signal=0
        it = {}
        log=[]
        utter=[]
        resp=[]
        con= row["Context"]
        if type=="tr":
            re=row["Utterance"]
        else:
            re=row["Ground Truth Utterance"]
        con =con.replace(". __eou__",".")
        con =con.replace("? __eou__", "?")
        con =con.replace("! __eou__", "!")
        con =con.replace("__eou__", ".")
        con=con.split("__eot__")[0:-1]
        re = re.replace(". __eou__",".")
        re = re.replace("? __eou__", "?")
        re =re.replace("__eou__", ".")
        con.append(re)
        if len(con)%2 ==1:
            con=con[0:-1]
        for i, c in enumerate(con):
            toklist = tok.encode(c)
            if len(toklist) > 400:
                print(len(toklist))
                signal = 1
            if i % 2 == 0:
                utter.append(c)
            else:
                resp.append(c)
        if signal==1:
            continue

        for i in range(len(utter)):
            dict = {}
            dict["user"] = utter[i]
            dict["user_delex"] = utter[i]
            toklist = tok.encode(utter[i])
            if len(toklist) > 400:
                continue
            dict["resp"] = resp[i]
            dict["nodelx_resp"] = resp[i]
            dict["pointer"] = "0,0,0,0,0,0"
            dict["match"] = ""
            dict["constraint"] = "[chit] [value_chit_act] chit"
            dict["cons_delex"] = "[chit] [value_chit_act]"
            dict["sys_act"] = "[chit] [chit_act] chit"
            dict["turn_num"] = i
            dict["turn_domain"] = "[chit]"
            log.append(dict)
        it["goal"] = {}
        it["log"]=log
        d["UB_"+str(index)]=it
    return d
tr=tran(train_data,"tr")
print(len(tr))
te=tran(test_data,"te")
save_json(tr, "./train_CC_UB.json")
save_json(te,"./test_CC_UB.json")
save_json(te,"./dev_CC_UB.json")




