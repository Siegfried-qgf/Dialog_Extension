'''
written by qgf 2022/3/28
'''
import spacy
import sys
import re
import string
from collections import Counter
sys.path.append("./../..")
from utils.io_utils import get_or_create_logger, load_json,save_json
import eval_QA
logger = get_or_create_logger(__name__)
nlp = spacy.load('en_core_web_sm')
class QA_evaluator(object):
    def __init__(self,data):
        self.data=load_json(data)

    def normalize_answer(self,s):
        """Lower text and remove punctuation, storys and extra whitespace."""

        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def get_tokens(self,s):
        if not s: return []
        return s.split()

    def compute_metric(self):
        count=0
        em=0
        f1=0
        for id,qas in self.data.items():
            for qa in qas:
                if qa["turn_num"]==0:
                    continue
                count+=1
                gold_data=self.normalize_answer(qa["resp"][11:-11])
                pred_data=self.normalize_answer(qa["resp_gen"][11:-11])
                #print(gold_data, pred_data)
                if gold_data==pred_data:
                    em+=1
                gold_toks = self.get_tokens(gold_data)
                pred_toks = self.get_tokens(pred_data)
                common = Counter(gold_toks) & Counter(pred_toks)
                num_same = sum(common.values())
                if len(gold_toks) == 0 or len(pred_toks) == 0:
                    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
                    f1+= int(gold_toks == pred_toks)
                elif num_same == 0:
                    f1+= 0
                else:
                    precision = 1.0 * num_same / len(pred_toks)
                    recall = 1.0 * num_same / len(gold_toks)
                    f1 += (2 * precision * recall) / (precision + recall)
                #print(em,f1)
        print(em/count,f1/count)




if __name__=="__main__":
    '''
    path="./QA_bsz4_ng1_aat_5e-4_5epoch_426_qadel/ckpt-epoch5/QA"
    results=load_json(path)
    rqa = []
    for dial_id, dial in results.items():
        for turn in dial:
            turn_dict = {}
            if turn['turn_num'] == 0:
                continue
            turn_dict['id'] = dial_id
            turn_dict["turn_id"] = turn['turn_num']
            turn_dict['answer'] = turn['resp_gen'][11:-11]
            rqa.append(turn_dict)
    save_json(rqa, "./outqa.json")
    f1 = eval_QA.eval_qa("./data/CQA/coqa-dev-v1.0.json", "outqa.json")
    print(f1)
    '''
    qa_eval=QA_evaluator("./QA_bsz4_ng1_aat_5e-4_5epoch_426_qadel/ckpt-epoch5/QA")
    qa_eval.compute_metric()


