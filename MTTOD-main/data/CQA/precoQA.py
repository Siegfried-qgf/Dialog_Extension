'''
by qgf 2022/3/18
'''
import json
import sys
sys.path.append("./../..")
from utils.io_utils import load_json,save_json
from transformers import T5Tokenizer
tok=T5Tokenizer.from_pretrained("t5-base")
def trans(path):
    data_m={}
    data=load_json(path)
    data=data['data']
    temp=0
    for item in data:
        it={}
        log=[]
        #add QA document embed
        dict={}
        '''
        toklist=tok.encode(item["story"])
        if len(toklist)>600:
            continue
        '''
        dict["user"] = "I wanna share a story with you. "+item["story"]
        dict["user_delex"] = "I wanna share a story with you. "+item["story"]
        dict["resp"] = "OK, let me see?"
        dict["nodelx_resp"] = "OK, let me see?"
        dict["pointer"] = "0,0,0,0,0,0"
        dict["match"] = ""
        dict["constraint"] = "[answer] [value_document] answer"
        dict["cons_delex"] = "[answer] [value_document]"
        dict["sys_act"] = "[answer] [document_act] answer"
        dict["turn_num"] = 0
        dict["turn_domain"] = "[answer]"
        log.append(dict)
        if temp<len(item["questions"]):
            temp=len(item["questions"])
        #for each QA turn
        for i in range(len(item["questions"])):
            dict={}
            dict["user"]=item["questions"][i]["input_text"]
            dict["user_delex"] = item["questions"][i]["input_text"]
            dict["resp"] = item["answers"][i]["input_text"]
            dict["nodelx_resp"] = item["answers"][i]["input_text"]
            dict["pointer"] ="0,0,0,0,0,0"
            dict["match"] =""
            dict["constraint"] = "[answer] [value_question] answer"
            dict["cons_delex"] = "[answer] [value_question]"
            dict["sys_act"] = "[answer] [answer_question] answer"
            dict["turn_num"] = i+1
            #todo 重新生成json
            dict["turn_domain"] = "[answer]"
            log.append(dict)

        it["goal"]={}
        it["log"]=log
        id=item["id"]
        data_m[id]=it
    print(len(data_m))
    save_json(data_m, "./dev_QA.json")
    return data_m


if __name__ == "__main__":
    trans('./coqa-dev-v1.0.json')

