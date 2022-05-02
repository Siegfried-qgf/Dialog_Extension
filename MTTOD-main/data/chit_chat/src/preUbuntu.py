'''
by qgf 2022/3/19
'''
import json
import sys
import tqdm
sys.path.append("./../../..")
from utils.io_utils import load_json,save_json,load_text
import pandas as pd

train_data=pd.read_csv("train.csv")
dev_data=pd.read_csv("valid.csv")
test_data=pd.read_csv("test.csv")
train_data=train_data[train_data["Label"]==1]
train_data=train_data.sample(n=100000)

def tran(data,type):
    d={}
    for index, row in tqdm.tqdm(data.iterrows()):
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
            if i % 2 == 0:
                utter.append(c)
            else:
                resp.append(c)

        for i in range(len(utter)):
            dict = {}
            dict["user"] = utter[i]
            dict["user_delex"] = utter[i]
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
        d["cc_"+str(index)]=it
    return d
tr=tran(train_data,"tr")
print(len(tr))
#te=tran(test_data,"te")
save_json(tr, "./train_CC.json")
#save_json(te,"./test_CC.json")




