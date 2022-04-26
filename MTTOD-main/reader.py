"""
   MTTOD: reader.py

   implements MultiWoz Training/Validation Data Feeder for MTTOD.

   This code is partially referenced from thu-spmi's damd-multiwoz repository:
   (https://github.com/thu-spmi/damd-multiwoz/blob/master/reader.py)

   Copyright 2021 ETRI LIRS, Yohan Lee
   Copyright 2019 Yichi Zhang

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import os
import copy
import spacy
import random
import difflib
from tqdm import tqdm
from difflib import get_close_matches
from itertools import chain
from collections import OrderedDict, defaultdict
from utils.io_utils import get_or_create_logger, load_json, save_json
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Tokenizer

from utils import definitions
from utils.io_utils import load_json, load_pickle, save_pickle, get_or_create_logger
from external_knowledges import MultiWozDB
import config
logger = get_or_create_logger(__name__)


class BaseIterator(object):
    def __init__(self, reader):
        self.reader = reader
        self.dial_by_domain = load_json("data/MultiWOZ_2.0/dial_by_domain.json")

#turn_bucket {对话轮数:对应dial}
    def bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []

            turn_bucket[turn_len].append(dial)

        return OrderedDict(sorted(turn_bucket.items(), key=lambda i: i[0]))

    def construct_mini_batch(self, data, batch_size, num_gpus):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            if len(batch) == batch_size:
                all_batches.append(batch)
                batch = []

        # if remainder > 1/2 batch_size, just put them in the previous batch, otherwise form a new batch
        if (len(batch) % num_gpus) != 0:
            batch = batch[:-(len(batch) % num_gpus)]
        if len(batch) > 0.5 * batch_size:
            all_batches.append(batch)
        elif len(all_batches):
            all_batches[-1].extend(batch)
        else:
            all_batches.append(batch)

        return all_batches

    def transpose_batch(self, dial_batch):
        turn_batch = []
        turn_num = len(dial_batch[0])
        for turn in range(turn_num):
            turn_l = []
            for dial in dial_batch:
                this_turn = dial[turn]
                turn_l.append(this_turn)
            turn_batch.append(turn_l)
        return turn_batch

    def get_batches(self, data_type, batch_size, num_gpus, shuffle=False, num_dialogs=-1, excluded_domains=None):
        dial = self.reader.data[data_type]
        # reader.data是已经encode的形式
        '''
                if excluded_domains is not None:
                    logger.info("Exclude domains: {}".format(excluded_domains))

                    target_dial_ids = []
                    for domains, dial_ids in self.dial_by_domain.items():
                        domain_list = domains.split("-")

                        if len(set(domain_list) & set(excluded_domains)) == 0:
                            target_dial_ids.extend(dial_ids)

                    dial = [d for d in dial if d[0]["dial_id"] in target_dial_ids]
        '''
        if num_dialogs > 0:
            dial = random.sample(dial, min(num_dialogs, len(dial)))

        turn_bucket = self.bucket_by_turn(dial)

        all_batches = []

        num_training_steps = 0
        num_turns = 0
        num_dials = 0
        for k in turn_bucket:
            #qa里最多轮数为36轮
            #if data_type != "test" and (k == 1 or k >= 17):
            #    continue
            if data_type != "test" and (k == 1 or k > 36):
               continue
            #test
            batches = self.construct_mini_batch(
                turn_bucket[k], batch_size, num_gpus)

            num_training_steps += k * len(batches)
            num_turns += k * len(turn_bucket[k])
            num_dials += len(turn_bucket[k])
            all_batches += batches

        if shuffle:
            random.shuffle(all_batches)

        return all_batches, num_training_steps, num_dials, num_turns

    def flatten_dial_history(self, dial_history, span_history, len_postfix, context_size):
        if context_size > 0:
            context_size -= 1

        if context_size == 0:
            windowed_context = []
            windowed_span_history = []
        elif context_size > 0:
            windowed_context = dial_history[-context_size:]
            windowed_span_history = span_history[-context_size:]
        else:
            windowed_context = dial_history
            windowed_span_history = span_history

        ctx_len = sum([len(c) for c in windowed_context])
        #print("here"+str(self.reader.max_seq_len))
        # consider eos_token
        #print('len_history'+str(ctx_len))
        #print("input"+str(len_postfix))
        spare_len = self.reader.max_seq_len - len_postfix - 1
        while ctx_len >= spare_len:

            ctx_len -= len(windowed_context[0])
            windowed_context.pop(0)
            if len(windowed_span_history) > 0:
                windowed_span_history.pop(0)

        context_span_info = defaultdict(list)
        for t, turn_span_info in enumerate(windowed_span_history):
            for domain, span_info in turn_span_info.items():
                if isinstance(span_info, dict):
                    for slot, spans in span_info.items():
                        adjustment = 0

                        if t > 0:
                            adjustment += sum([len(c)
                                            for c in windowed_context[:t]])

                        for span in spans:
                            start_idx = span[0] + adjustment
                            end_idx = span[1] + adjustment

                            context_span_info[slot].append((start_idx, end_idx))

                elif isinstance(span_info, list):
                    slot = domain
                    spans = span_info

                    adjustment = 0
                    if t > 0:
                        adjustment += sum([len(c)
                                           for c in windowed_context[:t]])

                    for span in spans:
                        start_idx = span[0] + adjustment
                        end_idx = span[1] + adjustment

                        context_span_info[slot].append((start_idx, end_idx))

        context = list(chain(*windowed_context))

        return context, context_span_info

    def tensorize(self, ids):
        return torch.tensor(ids, dtype=torch.long)

    def get_data_iterator(self, all_batches, task, ururu, add_auxiliary_task=False, context_size=-1):
        raise NotImplementedError


class MultiWOZIterator(BaseIterator):
    def __init__(self, reader):
        super(MultiWOZIterator, self).__init__(reader)
        self.num = 0

    def get_readable_batch(self, dial_batch):
        dialogs = {}

        decoded_keys = ["user","usdx", "resp", "redx", "bspn", "aspn", "dbpn",
                        "bspn_gen", "bspn_gen_with_span",
                        "dbpn_gen", "aspn_gen", "resp_gen"]
        for dial in dial_batch:
            dial_id = dial[0]["dial_id"]

            dialogs[dial_id] = []

            for turn in dial:
                readable_turn = {}

                for k, v in turn.items():
                    if k == "dial_id":
                        continue
                    elif k in decoded_keys:
                        v = self.reader.tokenizer.decode(
                            v, clean_up_tokenization_spaces=False)
                        '''
                        if k == "user":
                            print(k, v)
                        '''
                    elif k == "pointer":
                        turn_doamin = turn["turn_domain"][-1]
                        v = self.reader.db.pointerBack(v, turn_doamin)
                    if k == "user_span" or k == "resp_span":
                        speaker = k.split("_")[0]
                        v_dict = {}
                        for domain, ss_dict in v.items():
                            v_dict[domain] = {}
                            for s, span in ss_dict.items():
                                v_dict[domain][s] = self.reader.tokenizer.decode(
                                    turn[speaker][span[0]: span[1]])
                        v = v_dict

                    readable_turn[k] = v

                dialogs[dial_id].append(readable_turn)

        return dialogs

    def get_data_iterator(self, all_batches, task, ururu, add_auxiliary_task=False, context_size=-1):
        for dial_batch in all_batches:
            batch_encoder_input_ids_1 = []
            batch_encoder_input_ids_2 = []

            batch_span_label_ids = []
            batch_belief_label_ids = []
            batch_resp_label_ids = []

            for dial in dial_batch:
                dial_encoder_input_ids_1 = []
                dial_encoder_input_ids_2 = []
                dial_span_label_ids = []
                dial_belief_label_ids = []
                dial_resp_label_ids = []

                dial_history = []
                span_history = []
                for turn in dial:
                    context, span_dict = self.flatten_dial_history(
                        dial_history, span_history, len(turn["user"]), context_size)
                    encoder_input_ids_1 = context + turn["user"]+ [self.reader.eos_token_id]
                    #修改4/26
                    encoder_input_ids_2 = context + turn["user"] + turn["dbpn"] + [self.reader.eos_token_id]

                    # add current span of user utterance
                    for domain, ss_dict in turn["user_span"].items():
                        for s, span in ss_dict.items():
                            start_idx = span[0] + len(context)
                            end_idx = span[1] + len(context)
                            span_dict[s].append((start_idx, end_idx))

                    span_label_tokens = ["O"] * len(encoder_input_ids_1)
                    for slot, spans in span_dict.items():
                        for span in spans:
                            for i in range(span[0], span[1]):
                                #span_label_tokens[i] = "I-{}".format(slot)
                                span_label_tokens[i] = slot
                            #span_label_tokens[span[0]] = "B-{}".format(slot)

                    span_label_ids = [
                        self.reader.span_tokens.index(t) for t in span_label_tokens]

                    bspn = turn["bspn"]

                    bspn_label = bspn

                    belief_label_ids = bspn_label + [self.reader.eos_token_id]
                    '''
                    if task == "e2e":
                        resp = turn["dbpn"] + turn["aspn"] + turn["redx"]

                    else:
                        resp = turn["dbpn"] + turn["aspn"] + turn["resp"]
                    '''
                    # 修改4/27
                    #resp = turn["dbpn"] + turn["aspn"] + turn["redx"]
                    resp =turn["aspn"] + turn["redx"]
                    resp_label_ids = resp + [self.reader.eos_token_id]

                    dial_encoder_input_ids_1.append(encoder_input_ids_1)
                    dial_encoder_input_ids_2.append(encoder_input_ids_2)
                    dial_span_label_ids.append(span_label_ids)
                    dial_belief_label_ids.append(belief_label_ids)
                    dial_resp_label_ids.append(resp_label_ids)

                    turn_span_info = {}
                    for domain, ss_dict in turn["user_span"].items():
                        for s, span in ss_dict.items():
                            if domain not in turn_span_info:
                                turn_span_info[domain] = {}

                            if s not in turn_span_info[domain]:
                                turn_span_info[domain][s] = []

                            turn_span_info[domain][s].append(span)

                    if task == "dst":
                        for domain, ss_dict in turn["resp_span"].items():
                            for s, span in ss_dict.items():
                                if domain not in turn_span_info:
                                    turn_span_info[domain] = {}

                                if s not in turn_span_info[domain]:
                                    turn_span_info[domain][s] = []

                                adjustment = len(turn["user"])

                                if not ururu:
                                    adjustment += (len(bspn) +
                                                len(turn["dbpn"]) + len(turn["aspn"]))

                                start_idx = span[0] + adjustment
                                end_idx = span[1] + adjustment

                                turn_span_info[domain][s].append((start_idx, end_idx))

                    if ururu:
                        if task == "dst":
                            turn_text = turn["user"] + turn["resp"]
                        else:
                            turn_text = turn["user"] + turn["redx"]
                    else:
                        if task == "dst":
                            turn_text = turn["user"] + bspn + \
                                turn["dbpn"] + turn["aspn"] + turn["resp"]
                        else:
                        #todo:train时history用resp
                            turn_text = turn["user"] + bspn + \
                                turn["dbpn"] + turn["aspn"] + turn["resp"]

                    dial_history.append(turn_text)
                    span_history.append(turn_span_info)

                batch_encoder_input_ids_1.append(dial_encoder_input_ids_1)
                batch_encoder_input_ids_2.append(dial_encoder_input_ids_2)
                batch_span_label_ids.append(dial_span_label_ids)
                batch_belief_label_ids.append(dial_belief_label_ids)
                batch_resp_label_ids.append(dial_resp_label_ids)

            # turn first

            batch_encoder_input_ids_1 = self.transpose_batch(batch_encoder_input_ids_1)
            batch_encoder_input_ids_2 = self.transpose_batch(batch_encoder_input_ids_2)
            batch_span_label_ids = self.transpose_batch(batch_span_label_ids)
            batch_belief_label_ids = self.transpose_batch(batch_belief_label_ids)
            batch_resp_label_ids = self.transpose_batch(batch_resp_label_ids)





            num_turns = len(batch_encoder_input_ids_1)

            tensor_encoder_input_ids_1 = []
            tensor_encoder_input_ids_2 = []
            tensor_span_label_ids = []
            tensor_belief_label_ids = []
            tensor_resp_label_ids = []
            for t in range(num_turns):
                tensor_encoder_input_ids_1 = [
                    self.tensorize(b) for b in batch_encoder_input_ids_1[t]]
                tensor_encoder_input_ids_2 = [
                    self.tensorize(b) for b in batch_encoder_input_ids_2[t]]
                tensor_span_label_ids = [
                    self.tensorize(b) for b in batch_span_label_ids[t]]
                tensor_belief_label_ids = [
                    self.tensorize(b) for b in batch_belief_label_ids[t]]
                tensor_resp_label_ids = [
                    self.tensorize(b) for b in batch_resp_label_ids[t]]

                tensor_encoder_input_ids_1 = pad_sequence(tensor_encoder_input_ids_1,
                                                        batch_first=True,
                                                        padding_value=self.reader.pad_token_id)

                tensor_encoder_input_ids_2 = pad_sequence(tensor_encoder_input_ids_2,
                                                          batch_first=True,
                                                          padding_value=self.reader.pad_token_id)

                tensor_span_label_ids = pad_sequence(tensor_span_label_ids,
                                                    batch_first=True,
                                                    padding_value=self.reader.pad_token_id)

                tensor_belief_label_ids = pad_sequence(tensor_belief_label_ids,
                                                       batch_first=True,
                                                       padding_value=self.reader.pad_token_id)

                tensor_resp_label_ids = pad_sequence(tensor_resp_label_ids,
                                                     batch_first=True,
                                                     padding_value=self.reader.pad_token_id)

                yield (tensor_encoder_input_ids_1,tensor_encoder_input_ids_2), (tensor_span_label_ids, tensor_belief_label_ids, tensor_resp_label_ids)


class BaseReader(object):
    def __init__(self, backbone,cfg):
        self.cfg=cfg
        self.nlp = spacy.load("en_core_web_sm")

        self.tokenizer = self.init_tokenizer(backbone)

        self.data_dir = self.get_data_dir()

        encoded_data_path = os.path.join(self.data_dir, "encoded_data.pkl")

        if os.path.exists(encoded_data_path):
            logger.info("Load encoded data from {}".format(encoded_data_path))

            self.data = load_pickle(encoded_data_path)

        else:
            logger.info("Encode data and save to {}".format(encoded_data_path))
            train = self.encode_data("train")
            dev = self.encode_data("dev")
            test = self.encode_data("test")

            self.data = {"train": train, "dev": dev, "test": test}

            save_pickle(self.data, encoded_data_path)

        span_tokens = [self.pad_token, "O"]
        for slot in definitions.EXTRACTIVE_SLOT:
            #span_tokens.append("B-{}".format(slot))
            #span_tokens.append("I-{}".format(slot))
            span_tokens.append(slot)

        self.span_tokens = span_tokens

    def get_data_dir(self):
        raise NotImplementedError

    def init_tokenizer(self, backbone):
        tokenizer = T5Tokenizer.from_pretrained(backbone)

        special_tokens = []
#tokenizer加入special tokens
        # add domains
        domains = definitions.ALL_DOMAINS + ["general"]
        for domain in sorted(domains):
            token = "[" + domain + "]"
            special_tokens.append(token)

        # add intents
        intents = list(set(chain(*definitions.DIALOG_ACTS.values())))
        for intent in sorted(intents):
            token = "[" + intent + "]"
            special_tokens.append(token)

        # add slots
        slots = list(set(definitions.ALL_INFSLOT + definitions.ALL_REQSLOT))

        for slot in sorted(slots):
            token = "[value_" + slot + "]"
            special_tokens.append(token)

        special_tokens.extend(definitions.SPECIAL_TOKENS)
       # print(special_tokens)

        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
       # print(tokenizer.encode("[taxi]"))
       #print(tokenizer.encode("[chit]"))
       # print(tokenizer.encode("<bos_belief>"))
        return tokenizer

    @property
    def pad_token(self):
        return self.tokenizer.pad_token

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def unk_token(self):
        return self.tokenizer.unk_token

    @property
    def max_seq_len(self):
        return self.tokenizer.model_max_length

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def get_token_id(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)

    def encode_text(self, text, bos_token=None, eos_token=None):
        tokens = text.split() if isinstance(text, str) else text

        assert isinstance(tokens, list)

        if bos_token is not None:
            if isinstance(bos_token, str):
                bos_token = [bos_token]

            tokens = bos_token + tokens

        if eos_token is not None:
            if isinstance(eos_token, str):
                eos_token = [eos_token]

            tokens = tokens + eos_token

        encoded_text = self.tokenizer.encode(" ".join(tokens))

        # except eos token
        if encoded_text[-1] == self.eos_token_id:
            encoded_text = encoded_text[:-1]

        return encoded_text

    def encode_data(self, data_type):
        raise NotImplementedError


class MultiWOZReader(BaseReader):
    def __init__(self, backbone, version,cfg):
        self.version = version
        self.db = MultiWozDB(os.path.join("data", "MultiWOZ_{}".format(self.version), "db"))
        #todo maxlength
        super(MultiWOZReader, self).__init__(backbone,cfg)
        self.tokenizer.model_max_length = 1800

    def get_data_dir(self):
        print(self.cfg.data_type)
        if self.cfg.data_type=='TOD':
            return os.path.join("data", "MultiWOZ_{}".format(self.version), "processed")
        elif self.cfg.data_type=='CC':
            return os.path.join("data", "chit_chat","FusedChat")
        elif self.cfg.data_type=='QA':
            return os.path.join("data", "CQA")
        elif self.cfg.data_type=="CRS":
            return os.path.join("data", "CRS","ReDial")
        else:
            return os.path.join("data", "multi_data")

    def encode_data(self, data_type):
        vocab = self.tokenizer.get_vocab()
        j = save_json(vocab, './vocab.json')
        #todo dir
        if data_type=="train":
            data = load_json(self.get_data_dir()+"/train_{}.json".format(self.cfg.data_type))
        elif data_type=="dev":
            data = load_json(self.get_data_dir()+"/dev_{}.json".format(self.cfg.data_type))
        else:
            data = load_json(self.get_data_dir()+"/test_{}.json".format(self.cfg.data_type))

        encoded_data = []
        for fn, dial in tqdm(data.items(), desc=data_type, total=len(data)):
            encoded_dial = []

            accum_constraint_dict = {}
            for t in dial["log"]:
                turn_constrain_dict = self.bspn_to_constraint_dict(t["constraint"])
                for domain, sv_dict in turn_constrain_dict.items():
                    if domain not in accum_constraint_dict:
                        accum_constraint_dict[domain] = {}

                    for s, v in sv_dict.items():
                        if s not in accum_constraint_dict[domain]:
                            accum_constraint_dict[domain][s] = []

                        accum_constraint_dict[domain][s].append(v)

            prev_bspn = ""
            for idx, t in enumerate(dial["log"]):
                enc = {}
                enc["dial_id"] = fn
                enc["turn_num"] = t["turn_num"]
                enc["turn_domain"] = t["turn_domain"].split()
                if t["turn_domain"]!= "[recommend]":
                    enc["pointer"] = [int(i) for i in t["pointer"].split(",")]

                target_domain = enc["turn_domain"][0] if len(enc["turn_domain"]) == 1 else enc["turn_domain"][1]

                target_domain = target_domain[1:-1]

                user_ids = self.encode_text(t["user"],
                                            bos_token=definitions.BOS_USER_TOKEN,
                                            eos_token=definitions.EOS_USER_TOKEN)

                enc["user"] = user_ids

                usdx_ids = self.encode_text(t["user_delex"],
                                            bos_token=definitions.BOS_USER_TOKEN,
                                            eos_token=definitions.EOS_USER_TOKEN)

                enc["usdx"]=usdx_ids
                resp_ids = self.encode_text(t["nodelx_resp"],
                                            bos_token=definitions.BOS_RESP_TOKEN,
                                            eos_token=definitions.EOS_RESP_TOKEN)
                redx_ids = self.encode_text(t["resp"],
                                            bos_token=definitions.BOS_RESP_TOKEN,
                                            eos_token=definitions.EOS_RESP_TOKEN)
                #4/22
                if enc["turn_domain"][0]=="[recommend]":
                    enc["resp"]=redx_ids
                    enc["resp_rec"]=resp_ids
                else:
                    enc["resp"] = resp_ids
                    enc["resp_rec"] = ""

                enc["redx"] = redx_ids #redx 是有占位符的
                enc["match"]=t["match"]
                constraint_dict = self.bspn_to_constraint_dict(t["constraint"])

                ordered_constraint_dict = OrderedDict()
                for domain, slots in definitions.INFORMABLE_SLOTS.items():
                    if domain not in constraint_dict:
                        continue

                    ordered_constraint_dict[domain] = OrderedDict()
                    for slot in slots:
                        if slot not in constraint_dict[domain]:
                            continue

                        value = constraint_dict[domain][slot]

                        ordered_constraint_dict[domain][slot] = value

                ordered_bspn = self.constraint_dict_to_bspn(ordered_constraint_dict)

                bspn_ids = self.encode_text(ordered_bspn,
                                            bos_token=definitions.BOS_BELIEF_TOKEN,
                                            eos_token=definitions.EOS_BELIEF_TOKEN)

                enc["bspn"] = bspn_ids
                aspn_ids = self.encode_text(t["sys_act"],
                                            bos_token=definitions.BOS_ACTION_TOKEN,
                                            eos_token=definitions.EOS_ACTION_TOKEN)

                enc["aspn"] = aspn_ids
                if t["turn_domain"]!= "[recommend]":
                    pointer = enc["pointer"][:-2]
                    if not any(pointer):
                        db_token = definitions.DB_NULL_TOKEN
                    else:
                        db_token = "[db_{}]".format(pointer.index(1))
                else:
                    db_token=t["pointer"]

                dbpn_ids = self.encode_text(db_token,
                                            bos_token=definitions.BOS_DB_TOKEN,
                                            eos_token=definitions.EOS_DB_TOKEN)

                enc["dbpn"] = dbpn_ids

                if (len(enc["user"]) == 0 or len(enc["resp"]) == 0 or
                        len(enc["redx"]) == 0 or len(enc["bspn"]) == 0 or
                        len(enc["aspn"]) == 0 or len(enc["dbpn"]) == 0):
                    raise ValueError(fn, idx)

                # NOTE: if curr_constraint_dict does not include span[domain][slot], remove span[domain][slot] ??

                user_span = self.get_span(
                    target_domain,
                    self.tokenizer.convert_ids_to_tokens(user_ids),
                    self.tokenizer.convert_ids_to_tokens(usdx_ids),
                    accum_constraint_dict)

                enc["user_span"] = user_span

                resp_span = self.get_span(
                    target_domain,
                    self.tokenizer.convert_ids_to_tokens(resp_ids),
                    self.tokenizer.convert_ids_to_tokens(redx_ids),
                    accum_constraint_dict)
                enc["resp_span"] = resp_span
                encoded_dial.append(enc)
                prev_bspn = t["constraint"]

            encoded_data.append(encoded_dial)

        return encoded_data

    def bspn_to_constraint_dict(self, bspn):
        bspn = bspn.split() if isinstance(bspn, str) else bspn

        constraint_dict = OrderedDict()
        domain, slot = None, None
        for token in bspn:
            if token == definitions.EOS_BELIEF_TOKEN:
                break

            if token.startswith("["):
                token = token[1:-1]

                if token in definitions.ALL_DOMAINS:
                    domain = token

                if token.startswith("value_"):
                    if domain is None:
                        continue

                    if domain not in constraint_dict:
                        constraint_dict[domain] = OrderedDict()

                    slot = token.split("_",1)[1]

                    constraint_dict[domain][slot] = []

            else:
                try:
                    if domain is not None and slot is not None:
                        constraint_dict[domain][slot].append(token)
                except KeyError:
                    continue

        for domain, sv_dict in constraint_dict.items():
            for s, value_tokens in sv_dict.items():
                constraint_dict[domain][s] = " ".join(value_tokens)

        return constraint_dict

    def constraint_dict_to_bspn(self, constraint_dict):
        tokens = []
        for domain, sv_dict in constraint_dict.items():
            tokens.append("[" + domain + "]")
            for s, v in sv_dict.items():
                tokens.append("[value_" + s + "]")
                tokens.extend(v.split())

        return " ".join(tokens)

    def get_span(self, domain, text, delex_text, constraint_dict):
        span_info = {}

        if domain not in constraint_dict:
            return span_info

        tokens = text.split() if isinstance(text, str) else text

        delex_tokens = delex_text.split() if isinstance(delex_text, str) else delex_text

        seq_matcher = difflib.SequenceMatcher()

        seq_matcher.set_seqs(tokens, delex_tokens)

        for opcode in seq_matcher.get_opcodes():
            tag, i1, i2, j1, j2 = opcode

            lex_tokens = tokens[i1: i2]
            delex_token = delex_tokens[j1: j2]

            if tag == "equal" or len(delex_token) != 1:
                continue

            delex_token = delex_token[0]

            if not delex_token.startswith("[value_"):
                continue

            slot = delex_token[1:-1].split("_")[1]

            if slot not in definitions.EXTRACTIVE_SLOT:
                continue

            value = self.tokenizer.convert_tokens_to_string(lex_tokens)

            if slot in constraint_dict[domain] and value in constraint_dict[domain][slot]:
                if domain not in span_info:
                    span_info[domain] = {}

                span_info[domain][slot] = (i1, i2)

        return span_info

    def bspn_to_db_pointer(self, bspn, turn_domain):
        constraint_dict = self.bspn_to_constraint_dict(bspn)

        matnums = self.db.get_match_num(constraint_dict)
        match_dom = turn_domain[0] if len(turn_domain) == 1 else turn_domain[1]
        match_dom = match_dom[1:-1] if match_dom.startswith("[") else match_dom
        match = matnums[match_dom]

        vector = self.db.addDBIndicator(match_dom, match)

        return vector

    def canonicalize_span_value(self, domain, slot, value, cutoff=0.6):
        ontology = self.db.extractive_ontology

        if domain not in ontology or slot not in ontology[domain]:
            return value

        candidates = ontology[domain][slot]

        matches = get_close_matches(value, candidates, n=1, cutoff=cutoff)

        if len(matches) == 0:
            return value
        else:
            return matches[0]
    #2022/4/1
    def replace_placeholder(self, res, dbresult):
        all_tihuan1 = ['[value_' + i + "]" for i in definitions.ALL_INFSLOT]
        all_tihuan2 = ['[value_' + i + "]" for i in definitions.ALL_REQSLOT]
        all_tihuan = all_tihuan1 + all_tihuan2 + ["[value_choice]"]
        all_tihuan = list(set(all_tihuan))

        for t in all_tihuan:
            if t in res:
                t1 = t[7:-1]
                if (t1 != "choice"):
                    if (t1 == "reference"):
                        res = res.replace(t, str(random.randint(100000, 999999)))
                    else:
                        if dbresult[0].get(t1):
                            res = res.replace(t, str(dbresult[0][t1]))
                        else:
                            print("user_span里没有该slot 快去改代码")
                else:
                    res = res.replace(t, str(len(dbresult)))

        return res

    def resp_gen2resp_tihuan(self,resp_gen,turn):
        # 替换占位符2022/3/25 qgf
        cons = self.bspn_to_constraint_dict(self.tokenizer.decode(turn["bspn_gen_with_span"]))
        domain = turn["turn_domain"]
        tihuan=self.tokenizer.decode(resp_gen)
        for d in domain:
            resp = self.tokenizer.decode(resp_gen)
            d = d[1:-1]
            if d == 'taxi':
                color = random.choice(self.db.dbs['taxi']['taxi_colors'])
                type = random.choice(self.db.dbs['taxi']['taxi_types'])
                phone = str(random.randint(100000, 999999))
                venues = [{"color": color, "type": type, "phone": phone, "car": color + '' + type}]
                tihuan = self.replace_placeholder(resp, venues)

            if d == 'police':
                venues = self.db.dbs['police']
                tihuan = self.replace_placeholder(resp, venues)

            # if d=='hospital': 测试集里好像没有hospital

            if d in ['restaurant', 'hotel', 'attraction', 'train']:
                # print(self.reader.tokenizer.decode(turn["user"]))
                # print(cons)
                if cons:
                    if d in cons:
                        venues = self.db.queryJsons(d, cons[d], return_name=False)
                        if venues:
                            # print(venues)
                            tihuan = self.replace_placeholder(resp, venues)
                            # print(resp_tihuan)
                        # 如果没找到，尝试从user_span里找
                        else:
                            if turn["user_span"].get(d):
                                venues = copy.deepcopy([turn["user_span"][d]])
                                for dict in venues:
                                    for key, value in dict.items():
                                        (s, e) = value
                                        dict[key] = self.tokenizer.decode(turn['user'][s: e])
                                # print(venues)

                                tihuan = self.replace_placeholder(resp, venues)
                                # print(resp_tihuan)
                            else:
                                print("user_span里没有该domain")
                    else:
                        continue
                else:
                    continue
        return tihuan
'''
cfg=config.get_config()
reader=MultiWOZReader(cfg.backbone,cfg.version,cfg)

print(a)
'''