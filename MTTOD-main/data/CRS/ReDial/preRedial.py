'''
by qgf 2022/4/6
'''
import numpy as np
from tqdm import tqdm
import pickle as pkl
import json
from nltk import word_tokenize
import re
from torch.utils.data.dataset import Dataset
import numpy as np
from copy import deepcopy
import warnings
import sys
sys.path.append("./../../..")
from utils.io_utils import load_json,save_json
class dataset(object):
    def __init__(self, filename):
        self.entity2entityId = pkl.load(open('entity2entityId.pkl', 'rb'))
        self.entity_max = len(self.entity2entityId)
        self.id2entity = pkl.load(open('id2entity.pkl', 'rb'))
        self.subkg = pkl.load(open('subkg.pkl', 'rb'))  # need not back process
        self.text_dict = pkl.load(open('text_dict.pkl', 'rb'))
        # self.word2index=json.load(open('word2index.json',encoding='utf-8'))

        f = open(filename, encoding='utf-8')
        self.data = []
        self.corpus = []
        i=1
        for line in f:
            lines = json.loads(line.strip())
            movielist=lines["movieMentions"]
            seekerid = lines["initiatorWorkerId"]
            recommenderid = lines["respondentWorkerId"]
            contexts = lines['messages']
            movies = lines['movieMentions']
            altitude = lines['respondentQuestions']
            initial_altitude = lines['initiatorQuestions']
            cases = self._context_reformulate(movielist,contexts, movies, altitude, initial_altitude, seekerid, recommenderid)
            id="rec_"+str(i)
            i+=1
            self.data.append({id:cases})
            #print({id:cases})
        print("data长度"+str(len(self.data)))

        # if 'train' in filename:

        # self.prepare_word2vec()
        self.word2index = json.load(open('word2index_redial.json', encoding='utf-8'))
        self.key2index = json.load(open('key2index_3rd.json', encoding='utf-8'))
        self.stopwords = set([word.strip() for word in open('stopwords.txt', encoding='utf-8')])

        # self.co_occurance_ext(self.data)
        # exit()

    def prepare_word2vec(self):
        import gensim
        model = gensim.models.word2vec.Word2Vec(self.corpus, size=300, min_count=1)
        model.save('word2vec_redial')
        word2index = {word: i + 4 for i, word in enumerate(model.wv.index2word)}
        # word2index['_split_']=len(word2index)+4
        # json.dump(word2index, open('word2index_redial.json', 'w', encoding='utf-8'), ensure_ascii=False)
        word2embedding = [[0] * 300] * 4 + [model[word] for word in word2index] + [[0] * 300]
        import numpy as np

        word2index['_split_'] = len(word2index) + 4
        json.dump(word2index, open('word2index_redial.json', 'w', encoding='utf-8'), ensure_ascii=False)

        print(np.shape(word2embedding))
        np.save('word2vec_redial.npy', word2embedding)

    def padding_w2v(self, sentence, max_length, transformer=True, pad=0, end=2, unk=3):
        vector = []
        concept_mask = []
        dbpedia_mask = []
        for word in sentence:
            vector.append(self.word2index.get(word, unk))
            # if word.lower() not in self.stopwords:
            concept_mask.append(self.key2index.get(word.lower(), 0))
            # else:
            #    concept_mask.append(0)
            if '@' in word:
                try:
                    entity = self.id2entity[int(word[1:])]
                    id = self.entity2entityId[entity]
                except:
                    id = self.entity_max
                dbpedia_mask.append(id)
            else:
                dbpedia_mask.append(self.entity_max)
        vector.append(end)
        concept_mask.append(0)
        dbpedia_mask.append(self.entity_max)

        if len(vector) > max_length:
            if transformer:
                return vector[-max_length:], max_length, concept_mask[-max_length:], dbpedia_mask[-max_length:]
            else:
                return vector[:max_length], max_length, concept_mask[:max_length], dbpedia_mask[:max_length]
        else:
            length = len(vector)
            return vector + (max_length - len(vector)) * [pad], length, \
                   concept_mask + (max_length - len(vector)) * [0], dbpedia_mask + (max_length - len(vector)) * [
                       self.entity_max]

    def padding_context(self, contexts, pad=0, transformer=True):
        vectors = []
        vec_lengths = []
        if transformer == False:
            if len(contexts) > self.max_count:
                for sen in contexts[-self.max_count:]:
                    vec, v_l = self.padding_w2v(sen, self.max_r_length, transformer)
                    vectors.append(vec)
                    vec_lengths.append(v_l)
                return vectors, vec_lengths, self.max_count
            else:
                length = len(contexts)
                for sen in contexts:
                    vec, v_l = self.padding_w2v(sen, self.max_r_length, transformer)
                    vectors.append(vec)
                    vec_lengths.append(v_l)
                return vectors + (self.max_count - length) * [[pad] * self.max_c_length], vec_lengths + [0] * (
                            self.max_count - length), length
        else:
            contexts_com = []
            for sen in contexts[-self.max_count:-1]:
                contexts_com.extend(sen)
                contexts_com.append('_split_')
            contexts_com.extend(contexts[-1])
            vec, v_l, concept_mask, dbpedia_mask = self.padding_w2v(contexts_com, self.max_c_length, transformer)
            return vec, v_l, concept_mask, dbpedia_mask, 0

    def response_delibration(self, response, unk='MASKED_WORD'):
        new_response = []
        for word in response:
            if word in self.key2index:
                new_response.append(unk)
            else:
                new_response.append(word)
        return new_response

    def data_process(self, is_finetune=False):
        data_set = []
        context_before = []
        for line in self.data:
            # if len(line['contexts'])>2:
            #    continue
            if is_finetune and line['contexts'] == context_before:
                continue
            else:
                context_before = line['contexts']
            context, c_lengths, concept_mask, dbpedia_mask, _ = self.padding_context(line['contexts'])
            response, r_length, _, _ = self.padding_w2v(line['response'], self.max_r_length)
            if False:
                mask_response, mask_r_length, _, _ = self.padding_w2v(self.response_delibration(line['response']),
                                                                      self.max_r_length)
            else:
                mask_response, mask_r_length = response, r_length
            assert len(context) == self.max_c_length
            assert len(concept_mask) == self.max_c_length
            assert len(dbpedia_mask) == self.max_c_length

            data_set.append(
                [np.array(context), c_lengths, np.array(response), r_length, np.array(mask_response), mask_r_length,
                 line['entity'],
                 line['movie'], concept_mask, dbpedia_mask, line['rec']])
        return data_set

    def co_occurance_ext(self, data):
        stopwords = set([word.strip() for word in open('stopwords.txt', encoding='utf-8')])
        keyword_sets = set(self.key2index.keys()) - stopwords
        movie_wordset = set()
        for line in data:
            movie_words = []
            if line['rec'] == 1:
                for word in line['response']:
                    if '@' in word:
                        try:
                            num = self.entity2entityId[self.id2entity[int(word[1:])]]
                            movie_words.append(word)
                            movie_wordset.add(word)
                        except:
                            pass
            line['movie_words'] = movie_words
        new_edges = set()
        for line in data:
            if len(line['movie_words']) > 0:
                before_set = set()
                after_set = set()
                co_set = set()
                for sen in line['contexts']:
                    for word in sen:
                        if word in keyword_sets:
                            before_set.add(word)
                        if word in movie_wordset:
                            after_set.add(word)
                for word in line['response']:
                    if word in keyword_sets:
                        co_set.add(word)

                for movie in line['movie_words']:
                    for word in list(before_set):
                        new_edges.add('co_before' + '\t' + movie + '\t' + word + '\n')
                    for word in list(co_set):
                        new_edges.add('co_occurance' + '\t' + movie + '\t' + word + '\n')
                    for word in line['movie_words']:
                        if word != movie:
                            new_edges.add('co_occurance' + '\t' + movie + '\t' + word + '\n')
                    for word in list(after_set):
                        new_edges.add('co_after' + '\t' + word + '\t' + movie + '\n')
                        for word_a in list(co_set):
                            new_edges.add('co_after' + '\t' + word + '\t' + word_a + '\n')
        f = open('co_occurance.txt', 'w', encoding='utf-8')
        f.writelines(list(new_edges))
        f.close()
        json.dump(list(movie_wordset), open('movie_word.json', 'w', encoding='utf-8'), ensure_ascii=False)
        print(len(new_edges))
        print(len(movie_wordset))

    def entities2ids(self, entities):
        return [self.entity2entityId[word] for word in entities]

    def detect_movie(self, sentence, movies):

        token_text = word_tokenize(sentence)
        num = 0
        token_text_com = []
        while num < len(token_text):
            if token_text[num] == '@' and num + 1 < len(token_text):
                token_text_com.append(token_text[num] + token_text[num + 1])
                num += 2
            else:
                token_text_com.append(token_text[num])
                num += 1
        movie_rec = []
        for word in token_text_com:
            if word[1:] in movies:
                movie_rec.append(word[1:])
        movie_rec_trans = []
        for movie in movie_rec:
            entity = self.id2entity[int(movie)]
            try:
                movie_rec_trans.append(self.entity2entityId[entity])
            except:
                pass
        return token_text_com, movie_rec_trans, sentence

    def _context_reformulate(self, movielist,context, movies, altitude, ini_altitude, s_id, re_id):
        last_id = None
        # perserve the list of dialogue
        context_list = []
        for message in context:
            entities = []
            try:
                for entity in self.text_dict[message['text']]:
                    try:
                        entities.append(self.entity2entityId[entity])
                        #print(entities)
                    except:
                        pass
            except:
                pass
            token_text, movie_rec,sentence= self.detect_movie(message['text'], movies)
            if len(context_list) == 0:
                context_dict = {'movielist':movielist, 'text': token_text, 'entity': entities + movie_rec, 'user': message['senderWorkerId'],
                                'movie': movie_rec,'sentence':sentence}
                context_list.append(context_dict)
                last_id = message['senderWorkerId']
                continue
            if message['senderWorkerId'] == last_id:
                context_list[-1]['text'] += token_text
                context_list[-1]['entity'] += entities + movie_rec
                context_list[-1]['movie'] += movie_rec
                context_list[-1]['sentence'] += ' '+sentence
            else:
                context_dict = {'movielist':movielist,'text': token_text, 'entity': entities + movie_rec,
                                'user': message['senderWorkerId'], 'movie': movie_rec,'sentence':sentence}
                context_list.append(context_dict)
                last_id = message['senderWorkerId']

        cases = []
        contexts = []
        entities_set = set()
        entities = []
        for context_dict in context_list:
            self.corpus.append(context_dict['text'])
            if context_dict['user'] == re_id and len(contexts) > 0:
                response = context_dict['text']
                response_sen=context_dict['sentence']
                movielist=context_dict['movielist']
                if len(context_dict['movie']) != 0:
                    for movie in context_dict['movie']:
                        # if movie not in entities_set:
                        cases.append(
                        {'movielist':movielist,'contexts': deepcopy(contexts), 'response': response_sen, 'entity': deepcopy(entities),
                             'movie': movie, 'rec': 1})
                    contexts.clear()
                else:
                    cases.append(
                        {'movielist':movielist,'contexts': deepcopy(contexts), 'response': response_sen, 'entity': deepcopy(entities), 'movie': 0,
                         'rec': 0})
                    contexts.clear()

                #contexts.append(context_dict['text'])
                for word in context_dict['entity']:
                    if word not in entities_set:
                        entities.append(word)
                        entities_set.add(word)

            elif context_dict['user'] == re_id and len(contexts)== 0:
                for word in context_dict['entity']:
                    if word not in entities_set:
                        entities.append(word)
                        entities_set.add(word)
                cases.append({'movielist': movielist, 'contexts':["hi !"] , 'response': context_dict['sentence'],'entity': deepcopy(entities), 'movie': 0,'rec': 0})

            else:
                contexts.append(context_dict['sentence'])
                for word in context_dict['entity']:
                    if word not in entities_set:
                        entities.append(word)
                        entities_set.add(word)
        return cases

    def tihuan(self,sentence,movielist):
        new=[]
        for word in sentence:
            if word[0]=='@':
                continue
            if word.isdigit():
                if int(word) in self.id2entity:
                    if type(self.id2entity[int(word)])==str:
                        new.append(self.id2entity[int(word)][29:-1])
                    else:
                        warnings.warn("none type 电影号为"+word)
                        new.append(movielist[word])
                else:
                    warnings.warn("movie对不上 movie号为@"+word)
            else:
                new.append(word)
        return new

    def trans(self,data,train):
        data_m={}
        if train:
            movie_rec_list = load_json("movie_rec_list_train.json")
        else:
            movie_rec_list=load_json("movie_rec_list_test.json")
        count=0
        for cases in tqdm(data):
            #print(cases)
            it={}
            log=[]
            for id,turns in cases.items():

                for i in range(len(turns)):
                    dict={}
                    user_s=""
                    for word in turns[i]['contexts']:
                        user_s=user_s+word+", "
                    user_s=self.tihuan(word_tokenize(user_s[:-2]),turns[i]['movielist'])
                    user_tihuan=""
                    for word in user_s:
                        user_tihuan=user_tihuan+word+" "

                    dict["user"] = user_tihuan

                    user_d=""
                    for word in turns[i]['contexts']:
                        user_d=user_d+word+", "

                    dict["user_delex"]=user_d[:-2]

                    resp_s = self.tihuan(word_tokenize(turns[i]['response']),turns[i]['movielist'])
                    resp_tihuan = ""
                    for word in resp_s:
                        resp_tihuan = resp_tihuan + word + " "

                    dict["resp"] = resp_tihuan
                    #print(dict["user"])
                    #print(dict["resp"])
                    dict["turn_num"] = i
                    dict["nodelx_resp"]=turns[i]['response']
                    dict["pointer"] = "[db_recommend] "

                    if dict["user"] == "hi ! " and dict["turn_num"] == 0:
                        print("here")
                    else:
                        if movie_rec_list[count]:
                            for j in range(len(movie_rec_list[count])):
                                dict["pointer"] += movie_rec_list[count][j] + " "
                        count = count + 1



                    dict["match"] = turns[i]['movie']
                    dict["constraint"] = "[recommend] "
                    if turns[i]["rec"] == 0:
                        dict["constraint"] += "[value_recommend_chit] chit "
                    if turns[i]['entity']:
                        dict["constraint"] += "[value_entity] "
                        movie_name=""
                        for entity in turns[i]['entity']:
                            for key, value in self.entity2entityId.items():
                                if value == entity:
                                    movie_name=key
                                    dict["constraint"]+=str(key)[29:-1]+" "
                            if movie_name == "":
                                warnings.warn("entitiy对应不上 entity号为"+str(entity))
                            movie_name=""


                    if turns[i]["rec"]==1:
                        dict["sys_act"] = "[recommend] [recommend_act] recommend"
                    else:
                        dict["sys_act"] = "[recommend] [recommend_chit] chit"

                    dict["turn_domain"] = "[recommend]"
                    log.append(dict)
                    #print(dict)
                it['goal']={}
                it["log"]=log
                print(count-1)
            #print({id:it})
            data_m.update({id:it})
        return data_m

class CRSdataset(Dataset):
    def __init__(self, dataset, entity_num, concept_num):
        self.data = dataset
        self.entity_num = entity_num
        self.concept_num = concept_num + 1

    def __getitem__(self, index):
        '''
        movie_vec = np.zeros(self.entity_num, dtype=np.float)
        context, c_lengths, response, r_length, entity, movie, concept_mask, dbpedia_mask, rec = self.data[index]
        for en in movie:
            movie_vec[en] = 1 / len(movie)
        return context, c_lengths, response, r_length, entity, movie_vec, concept_mask, dbpedia_mask, rec
        '''
        context, c_lengths, response, r_length, mask_response, mask_r_length, entity, movie, concept_mask, dbpedia_mask, rec = \
        self.data[index]
        entity_vec = np.zeros(self.entity_num)
        entity_vector = np.zeros(50, dtype=np.int)
        point = 0
        for en in entity:
            entity_vec[en] = 1
            entity_vector[point] = en
            point += 1

        concept_vec = np.zeros(self.concept_num)
        for con in concept_mask:
            if con != 0:
                concept_vec[con] = 1

        db_vec = np.zeros(self.entity_num)
        for db in dbpedia_mask:
            if db != 0:
                db_vec[db] = 1

        return context, c_lengths, response, r_length, mask_response, mask_r_length, entity_vec, entity_vector, movie, np.array(
            concept_mask), np.array(dbpedia_mask), concept_vec, db_vec, rec

    def __len__(self):
        return len(self.data)




if __name__ == '__main__':

    ds = dataset('train_data.jsonl')
    save_json(ds.trans(ds.data,True),"./train_CRS.json")
    '''
    ds=dataset("test_data.jsonl")
    save_json(ds.trans(ds.data, False), "./test_CRS.json")
    '''




