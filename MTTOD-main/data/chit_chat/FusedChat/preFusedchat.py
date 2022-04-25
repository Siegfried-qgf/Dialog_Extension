'''
by qgf 2022/3/19
'''
import json
import sys
sys.path.append("./../../..")
from utils.io_utils import load_json,save_json,load_text

def chit_chat_list(turns,types):
    cclist=[]
    for i in range(len(turns)):
        if types[i]=="appended" or types[i]=="prepended":
            cclist.append(turns[i])
    return cclist

def list2pair(cclist):
    pair=[]
    for i in range(int(len(cclist)/2)):
        pair.append((cclist[i],cclist[i+1]))
    return pair

def trans(path):
    data_m_train={}
    data_m_dev={}
    data_m_test={}
    data=load_json(path,lower=False)#dict{ id :{turns:[] types:[]}}
    testlist=load_text("./testListFile.json",False)
    vallist=load_text("./valListFile.json",False)
    for item in data.items():
        it={}
        log=[]
        (id,subitem)=item
        cclist=chit_chat_list(subitem["turns"],subitem["types"])
        pairs=list2pair(cclist)
        for i in range(len(pairs)):
            (user,sys)=pairs[i]
            dict={}
            dict["user"] = user
            dict["user_delex"] = user
            dict["resp"] = sys
            dict["nodelx_resp"] = sys
            dict["pointer"] = "0,0,0,0,0,0"
            dict["match"] = ""
            dict["constraint"] = "[chit] [value_chit_act] chit"
            dict["cons_delex"] = "[chit] [value_chit_act]"
            dict["sys_act"] = "[chit] [chit_act] chit"
            dict["turn_num"] = i
            dict["turn_domain"] = "[chit]"
            log.append(dict)
        it["goal"]={}
        it["log"] = log
        id_ext=id+".json"
        if id_ext in testlist:
            id=id+"_cc"
            data_m_test[id]=it
        elif id_ext in vallist:
            id = id + "_cc"
            data_m_dev[id]=it
        else:
            id = id + "_cc"
            data_m_train[id]=it
    return data_m_train,data_m_dev,data_m_test


'''
def count():
    data=load_json("./train_CC.json",lower=False)
    print("train"+str(len(data)))
    data = load_json("./dev_CC.json",lower=False)
    print("dev" + str(len(data)))
    data = load_json("./test_CC.json",lower=False)
    print("test" + str(len(data)))
'''
if __name__ == "__main__":
    tr_a,de_a,te_a=trans("./appended.json")
    tr_p, de_p, te_p = trans("./prepended.json")
    tr_a.update(tr_p)
    de_a.update(de_p)
    te_a.update(te_p)
    save_json(tr_a,"./train_CC.json")
    save_json(de_a, "./dev_CC.json")
    save_json(te_a, "./test_CC.json")

