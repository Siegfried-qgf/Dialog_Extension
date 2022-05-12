'''
written by qgf 2022/4/22
'''
import tqdm
import pickle as pkl

entity2entityid = pkl.load(open("./data/CRS/ReDial/entity2entityId.pkl", "rb"))
def id2movie(v):
    for key, value in entity2entityid.items():
        if value == v:
            return key[29:-1]
    return ""

def recall(data):
    count=0
    recall_1=0
    for dial_id, dial in tqdm.tqdm(data.items()):
        for turn in dial:
            if turn["match"]:
                count=count+1
                gold=id2movie(turn["match"])
                gold=gold.lower()
                if gold=="":
                    continue
                if gold in turn["resp_gen"]:
                    recall_1=recall_1+1
    print("匹配到的 "+str(recall_1))
    print("总数 "+str(count))
    return recall_1/count

#crs=load_json("./MUL_5.6/ckpt-epoch5/CRS_goal")
#print(recall(crs))







