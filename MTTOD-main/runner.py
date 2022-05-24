import os
import re
import copy
import math
import time
import glob
import shutil
import json
from abc import *
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import sys
sys.path.append("./..")
import numpy as np
import pickle as pkl
import warnings
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import sampler,DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from transformers.modeling_outputs import BaseModelOutput
from tensorboardX import SummaryWriter
from utils.io_utils import load_json,save_json
from model import T5WithSpan, T5WithTokenSpan
from reader import MultiWOZIterator, MultiWOZReader,MultiWOZDataset, SequentialDistributedSampler, CollatorTrain
from evaluator import MultiWozEvaluator
from nltk import word_tokenize
from utils import definitions,MyDataLoader
from utils.io_utils import get_or_create_logger, load_json, save_json ,gen_mask_with_prob
from utils.ddp_utils import reduce_mean, reduce_sum
import eval_chitchat
import eval_CRS
import eval_QA
logger = get_or_create_logger(__name__)

class Reporter(object):
    def __init__(self, log_frequency, model_dir):
        self.log_frequency = log_frequency
        self.summary_writer = SummaryWriter(os.path.join(model_dir, "tensorboard"))

        self.global_step = 0
        self.lr = 0
        self.init_stats()

    def init_stats(self):
        self.step_time = 0.0

        self.belief_loss = 0.0
        self.span_loss = 0.0
        self.resp_loss = 0.0

        self.belief_correct = 0.0
        self.span_correct = 0.0
        self.resp_correct = 0.0

        self.belief_count = 0.0
        self.span_count = 0.0
        self.resp_count = 0.0

    def step(self, start_time, lr, step_outputs, force_info=False, is_train=True):
        self.global_step += 1
        self.step_time += (time.time() - start_time)

        self.belief_loss += step_outputs["belief"]["loss"]
        self.belief_correct += step_outputs["belief"]["correct"]
        self.belief_count += step_outputs["belief"]["count"]

        if "span" in step_outputs:
            self.span_loss += step_outputs["span"]["loss"]
            self.span_correct += step_outputs["span"]["correct"]
            self.span_count += step_outputs["span"]["count"]

            do_span_stats = True
        else:
            do_span_stats = False

        if "resp" in step_outputs:
            self.resp_loss += step_outputs["resp"]["loss"]
            self.resp_correct += step_outputs["resp"]["correct"]
            self.resp_count += step_outputs["resp"]["count"]

            do_resp_stats = True
        else:
            do_resp_stats = False

        if is_train:
            self.lr = lr
            self.summary_writer.add_scalar("lr", lr, global_step=self.global_step)

            if self.global_step % self.log_frequency == 0:
                self.info_stats("train", self.global_step, do_span_stats, do_resp_stats)

    def info_stats(self, data_type, global_step, do_span_stats=False, do_resp_stats=False):
        avg_step_time = self.step_time / self.log_frequency

        belief_ppl = math.exp(self.belief_loss / self.belief_count)
        belief_acc = (self.belief_correct / self.belief_count) * 100

        self.summary_writer.add_scalar(
            "{}/belief_loss".format(data_type), self.belief_loss, global_step=global_step)

        self.summary_writer.add_scalar(
            "{}/belief_ppl".format(data_type), belief_ppl, global_step=global_step)

        self.summary_writer.add_scalar(
            "{}/belief_acc".format(data_type), belief_acc, global_step=global_step)

        if data_type == "train":
            common_info = "step {0:d}; step-time {1:.2f}s; lr {2:.2e};".format(
                global_step, avg_step_time, self.lr)
        else:
            common_info = "[Validation]"

        belief_info = "[belief] loss {0:.2f}; ppl {1:.2f}; acc {2:.2f}".format(
            self.belief_loss, belief_ppl, belief_acc)

        if do_resp_stats:
            resp_ppl = math.exp(self.resp_loss / self.resp_count)
            resp_acc = (self.resp_correct / self.resp_count) * 100

            self.summary_writer.add_scalar(
                "{}/resp_loss".format(data_type), self.resp_loss, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/resp_ppl".format(data_type), resp_ppl, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/resp_acc".format(data_type), resp_acc, global_step=global_step)

            resp_info = "[resp] loss {0:.2f}; ppl {1:.2f}; acc {2:.2f}".format(
                self.resp_loss, resp_ppl, resp_acc)
        else:
            resp_info = ""

        if do_span_stats:
            if self.span_count == 0:
                span_acc = 0.0
            else:
                span_acc = (self.span_correct / self.span_count) * 100

            self.summary_writer.add_scalar(
                "{}/span_loss".format(data_type), self.span_loss, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/span_acc".format(data_type), span_acc, global_step=global_step)

            span_info = "[span] loss {0:.2f}; acc {1:.2f};".format(
                self.span_loss, span_acc)

        else:
            span_info = ""

        logger.info(
            " ".join([common_info, belief_info, resp_info, span_info]))

        self.init_stats()

class BaseRunner(metaclass=ABCMeta):
    def __init__(self, cfg, reader):
        self.cfg = cfg
        self.reader = reader
        self.model = self.load_model()

    def load_model(self):
        if self.cfg.ckpt is not None:
            model_path = self.cfg.ckpt
            initialize_additional_decoder = False
        elif self.cfg.train_from is not None:
            model_path = self.cfg.train_from
            initialize_additional_decoder = False
        else:
            model_path = self.cfg.backbone
            initialize_additional_decoder = True
        logger.info("Load model from {}".format(model_path))
        #判断是否加辅助任务
        if not self.cfg.add_auxiliary_task:
            model_wrapper = T5WithSpan
        else:
            model_wrapper = T5WithTokenSpan

        #aux任务的分类数
        num_span = len(definitions.EXTRACTIVE_SLOT)

        model = model_wrapper.from_pretrained(model_path, num_span=num_span)

        model.resize_token_embeddings(self.reader.vocab_size)

        if initialize_additional_decoder:
            model.initialize_additional_decoder()

        model.to(self.cfg.device)

        if self.cfg.num_gpus > 1:
            model=nn.parallel.DistributedDataParallel(model,device_ids=[self.cfg.local_rank], output_device=self.cfg.local_rank)

        model.to(self.cfg.device)

        return model

    def save_model(self, epoch):
        latest_ckpt = "ckpt-epoch{}".format(epoch)
        save_path = os.path.join(self.cfg.model_dir, latest_ckpt)

        #todo: dp
        if self.cfg.num_gpus > 1:
            model = self.model.module
        else:
            model = self.model



        model.save_pretrained(save_path)

        # keep chekpoint up to maximum
        checkpoints = sorted(
            glob.glob(os.path.join(self.cfg.model_dir, "ckpt-*")),
            key=os.path.getmtime,
            reverse=True)

        checkpoints_to_be_deleted = checkpoints[self.cfg.max_to_keep_ckpt:]

        for ckpt in checkpoints_to_be_deleted:
            shutil.rmtree(ckpt)

        return latest_ckpt

    def get_optimizer_and_scheduler(self, num_traininig_steps_per_epoch, train_batch_size):
        '''
        num_train_steps = (num_train_examples *
            self.cfg.epochs) // (train_batch_size * self.cfg.grad_accum_steps)
        '''
        num_train_steps = (num_traininig_steps_per_epoch *
            self.cfg.epochs) // self.cfg.grad_accum_steps

        if self.cfg.warmup_steps >= 0:
            num_warmup_steps = self.cfg.warmup_steps
        else:
            #num_warmup_steps = int(num_train_steps * 0.2)
            num_warmup_steps = int(num_train_steps * self.cfg.warmup_ratio)

        logger.info("Total training steps = {}, warmup steps = {}".format(
            num_train_steps, num_warmup_steps))

        optimizer = AdamW(self.model.parameters(), lr=self.cfg.learning_rate)

        if self.cfg.no_learning_rate_decay:
            scheduler = get_constant_schedule(optimizer)
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_train_steps)

        return optimizer, scheduler

    def count_tokens(self, pred, label, pad_id):
        pred = pred.view(-1)
        label = label.view(-1)

        num_count = label.ne(pad_id).long().sum()
        num_correct = torch.eq(pred, label).long().sum()

        return num_correct, num_count

    def count_spans(self, pred, label):
        pred = pred.view(-1, 2)

        num_count = label.ne(-1).long().sum()
        num_correct = torch.eq(pred, label).long().sum()

        return num_correct, num_count

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError

class MultiWOZRunner(BaseRunner):
    def __init__(self, cfg):
        reader = MultiWOZReader(cfg.backbone, cfg.version, cfg)
        opt={'max_c_length': 256, 'max_r_length': 30, 'batch_size': 32, 'max_count': 5, 'use_cuda': True, 'load_dict': None,
         'learningrate': 0.001, 'optimizer': 'adam', 'momentum': 0, 'is_finetune': False, 'embedding_type': 'random',
         'epoch': 30, 'gpu': '0,1', 'gradient_clip': 0.1, 'embedding_size': 300, 'n_heads': 2, 'n_layers': 2,
         'ffn_size': 300, 'dropout': 0.1, 'attention_dropout': 0.0, 'relu_dropout': 0.1,
         'learn_positional_embeddings': False, 'embeddings_scale': True, 'n_entity': 64368, 'n_relation': 214,
         'n_concept': 29308, 'n_con_relation': 48, 'dim': 128, 'n_hop': 2, 'kge_weight': 1, 'l2_weight': 2.5e-06,
         'n_memory': 32, 'item_update_mode': '0,1', 'using_all_hops': True, 'num_bases': 8}
        if cfg.data_type=="CRS" and cfg.run_type=="predict":
        #if cfg.data_type=="MUL_T":
            from KGSF import run
            self.loop=run.TrainLoop_fusion_rec(opt,is_finetune=False)
            self.loop.model.load_model()
            self.entity2entityId = pkl.load(open('./data/CRS/ReDial/entity2entityId.pkl', 'rb'))
            self.e2id = {}
            a = list(self.entity2entityId.keys())
            for k in a:
                temp = self.entity2entityId[k]
                if isinstance(k, str):
                    k = k.lower()
                    self.e2id[k] = temp
            self.entity_max = len(self.entity2entityId)
            self.id2entity = pkl.load(open('./data/CRS/ReDial/id2entity.pkl', 'rb'))
            self.subkg = pkl.load(open('./data/CRS/ReDial/subkg.pkl', 'rb'))  # need not back process
            self.text_dict = pkl.load(open('./data/CRS/ReDial/text_dict.pkl', 'rb'))
            self.word2index = json.load(open('./data/CRS/ReDial/word2index_redial.json', encoding='utf-8'))
            self.key2index = json.load(open('./data/CRS/ReDial/key2index_3rd.json', encoding='utf-8'))
            self.stopwords = set(
                        [word.strip() for word in open('./data/CRS/ReDial/stopwords.txt', encoding='utf-8')])
            self.movie_ids = pkl.load(open("./data/CRS/ReDial/movie_ids.pkl", "rb"))
        self.iterator = MultiWOZIterator(reader)

        super(MultiWOZRunner, self).__init__(cfg, reader)

    def step_fn(self, inputs, span_labels, belief_labels, resp_labels):
        inputs_1,inputs_2=inputs
        inputs_1 = inputs_1.to(self.cfg.device)
        inputs_2 = inputs_2.to(self.cfg.device)
        span_labels = span_labels.to(self.cfg.device)
        belief_labels = belief_labels.to(self.cfg.device)
        resp_labels = resp_labels.to(self.cfg.device)

        attention_mask_1 = torch.where(inputs_1 == self.reader.pad_token_id, 0, 1)

        belief_outputs = self.model(input_ids=inputs_1,
                                    attention_mask=attention_mask_1,
                                    span_labels=span_labels,
                                    lm_labels=belief_labels,
                                    return_dict=False,
                                    add_auxiliary_task=self.cfg.add_auxiliary_task,
                                    decoder_type="belief")

        belief_loss = belief_outputs[0]
        belief_pred = belief_outputs[1]

        span_loss = belief_outputs[2]
        span_pred = belief_outputs[3]


        if self.cfg.task == "e2e":
            attention_mask_2 = torch.where(inputs_2 == self.reader.pad_token_id, 0, 1)
            #last_hidden_state = belief_outputs[5]

            #encoder_outputs = BaseModelOutput(last_hidden_state=last_hidden_state)

            #保存之前参数
            #当前para*mask

            resp_outputs = self.model(input_ids=inputs_2,
                                      attention_mask=attention_mask_2,
                                      lm_labels=resp_labels,
                                      return_dict=False,
                                      decoder_type="resp")

            resp_loss = resp_outputs[0]
            resp_pred = resp_outputs[1]

            num_resp_correct, num_resp_count = self.count_tokens(
                resp_pred, resp_labels, pad_id=self.reader.pad_token_id)

        num_belief_correct, num_belief_count = self.count_tokens(
            belief_pred, belief_labels, pad_id=self.reader.pad_token_id)

        if self.cfg.add_auxiliary_task:
            num_span_correct, num_span_count = self.count_tokens(
                span_pred, span_labels, pad_id=0)

        loss = belief_loss

        if self.cfg.add_auxiliary_task and self.cfg.aux_loss_coeff > 0:
            loss += (self.cfg.aux_loss_coeff * span_loss)

        if self.cfg.task == "e2e" and self.cfg.resp_loss_coeff > 0:
            loss += (self.cfg.resp_loss_coeff * resp_loss)

        step_outputs = {"belief": {"loss": belief_loss.item(),
                                   "correct": num_belief_correct.item(),
                                   "count": num_belief_count.item()}}

        if self.cfg.add_auxiliary_task:
            step_outputs["span"] = {"loss": span_loss.item(),
                                    "correct": num_span_correct.item(),
                                    "count": num_span_count.item()}

        if self.cfg.task == "e2e":
            step_outputs["resp"] = {"loss": resp_loss.item(),
                                    "correct": num_resp_correct.item(),
                                    "count": num_resp_count.item()}

        return loss, step_outputs

    def step_fn_bs(self, inputs, span_labels, belief_labels):
        inputs_1,_ = inputs
        inputs_1 = inputs_1.to(self.cfg.device)
        span_labels = span_labels.to(self.cfg.device)
        belief_labels = belief_labels.to(self.cfg.device)

        attention_mask_1 = torch.where(inputs_1 == self.reader.pad_token_id, 0, 1)

        belief_outputs = self.model(input_ids=inputs_1,
                                    attention_mask=attention_mask_1,
                                    span_labels=span_labels,
                                    lm_labels=belief_labels,
                                    return_dict=False,
                                    add_auxiliary_task=self.cfg.add_auxiliary_task,
                                    decoder_type="belief")

        belief_loss = belief_outputs[0]
        belief_pred = belief_outputs[1]

        span_loss = belief_outputs[2]
        span_pred = belief_outputs[3]

        num_belief_correct, num_belief_count = self.count_tokens(
            belief_pred, belief_labels, pad_id=self.reader.pad_token_id)

        if self.cfg.add_auxiliary_task:
            num_span_correct, num_span_count = self.count_tokens(
                span_pred, span_labels, pad_id=0)

        loss = belief_loss
        step_outputs = {"belief": {"loss": belief_loss.item(),
                                   "correct": num_belief_correct.item(),
                                   "count": num_belief_count.item()}}

        if self.cfg.add_auxiliary_task and self.cfg.aux_loss_coeff > 0:
            loss += (self.cfg.aux_loss_coeff * span_loss)
            step_outputs["span"] = {"loss": span_loss.item(),
                                    "correct": num_span_correct.item(),
                                    "count": num_span_count.item()}

        return loss,step_outputs

    def step_fn_resp(self,inputs,resp_labels):
        _, inputs_2 = inputs
        inputs_2 = inputs_2.to(self.cfg.device)
        resp_labels = resp_labels.to(self.cfg.device)

        attention_mask_2 = torch.where(inputs_2 == self.reader.pad_token_id, 0, 1)

        resp_outputs = self.model(input_ids=inputs_2,
                                  attention_mask=attention_mask_2,
                                  lm_labels=resp_labels,
                                  return_dict=False,
                                  decoder_type="resp")
        resp_loss = resp_outputs[0]
        resp_pred = resp_outputs[1]

        num_resp_correct, num_resp_count = self.count_tokens(
            resp_pred, resp_labels, pad_id=self.reader.pad_token_id)

        loss=resp_loss
        step_outputs = {"resp": {"loss": resp_loss.item(),
                                    "correct": num_resp_correct.item(),
                                    "count": num_resp_count.item()}}
        return loss, step_outputs

    def reduce_ddp_stepoutpus(self, step_outputs):
        step_outputs_all = {"belief": {"loss": reduce_mean(step_outputs['belief']['loss']),
                            "correct": reduce_sum(step_outputs['belief']['correct']),
                            "count": reduce_sum(step_outputs['belief']['count'])}}

        if self.cfg.add_auxiliary_task:
            step_outputs_all['span'] = {
                'loss': reduce_mean(step_outputs['span']['loss']),
                "correct": reduce_sum(step_outputs['span']['correct']),
                "count": reduce_sum(step_outputs['span']['count'])
            }

        if self.cfg.task == "e2e":
            step_outputs_all["resp"] = {
                'loss': reduce_mean(step_outputs['resp']['loss']),
                "correct": reduce_sum(step_outputs['resp']['correct']),
                "count": reduce_sum(step_outputs['resp']['count'])
            }

        return step_outputs_all

    def train_epoch(self, data_loader, optimizer, scheduler, reporter=None):
        self.model.train()
        self.model.zero_grad()
        epoch_step, train_loss = 0, 0.
        for batch in iter(data_loader):
            start_time_bs = time.time()
            inputs, span_labels, belief_labels, resp_labels,turn_type = batch

            #loss, step_outputs = self.step_fn(inputs, span_labels, belief_labels, resp_labels)
            loss_bs,step_outputs_bs=self.step_fn_bs(inputs,span_labels,belief_labels)

            if self.cfg.grad_accum_steps > 1:
                loss_bs = loss_bs / self.cfg.grad_accum_steps

            loss_bs.backward()
            train_loss += loss_bs.item()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_grad_norm)

            if (epoch_step + 1) % self.cfg.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                lr = scheduler.get_last_lr()[0]
                if reporter is not None:
                    reporter.step(start_time_bs, lr, step_outputs_bs)

            if self.cfg.train_subnet:
                # dst结束 保存参数
                pre_dict=copy.deepcopy(self.model.state_dict())

                if turn_type[0] == 0:
                    self.mask_dict=self.mask_dict_tod
                elif turn_type[0]== 1 or turn_type[0]==2:
                    self.mask_dict=self.mask_dict_cc
                elif turn_type[0]==3 or turn_type[0]==4:
                    self.mask_dict=self.mask_dict_qa
                elif turn_type[0]==5:
                    self.mask_dict=self.mask_dict_crs
                else:
                    warnings.warn("找不到该类型"+turn_type[0])
                for k, v in self.model.named_parameters():
                    if k in self.mask_dict:
                        v.data = v * self.mask_dict[k].cuda()

            start_time_resp = time.time()
            loss_resp, step_outputs_resp = self.step_fn_resp(inputs, resp_labels)

            if self.cfg.grad_accum_steps > 1:
                loss_resp= loss_resp / self.cfg.grad_accum_steps

            loss_resp.backward()
            train_loss+=loss_resp.item()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_grad_norm)

            if (epoch_step + 1) % self.cfg.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                lr = scheduler.get_last_lr()[0]
                if reporter is not None:
                    reporter.step(start_time_resp, lr, step_outputs_resp)
            if self.cfg.train_subnet:
                for k, v in self.mask_dict:
                    dim0,dim1=v.shape
                    for i in range(dim0):
                        for j in range(dim1):
                            if v[i][j]==False:
                                self.model.state_dict()[k][i][j]=pre_dict[k][i][j]
                            else:
                                continue
            epoch_step += 1

        return train_loss

    '''
    def train_epoch(self, data_loader, optimizer, scheduler, reporter=None):
        self.model.train()
        self.model.zero_grad()
        epoch_step, train_loss = 0, 0.
        for batch in iter(data_loader):
            start_time = time.time()

            inputs, span_labels, belief_labels, resp_labels,_ = batch

            loss, step_outputs = self.step_fn(inputs, span_labels, belief_labels, resp_labels)

            if self.cfg.grad_accum_steps > 1:
                loss = loss / self.cfg.grad_accum_steps

            loss.backward()
            train_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_grad_norm)

            if (epoch_step + 1) % self.cfg.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                lr = scheduler.get_last_lr()[0]

                if reporter is not None:
                    reporter.step(start_time, lr, step_outputs)

            epoch_step += 1

        return train_loss
    '''
    def train(self):
        if self.cfg.train_subnet:
            self.mask_dict_tod=torch.load("./ckpt/TOD_only_epoch10/ckpt-epoch8/subnet.pth")
            self.mask_dict_cc=torch.load("./ckpt/CC_UB_FU_epoch10/ckpt-epoch10/subnet.pth")
            self.mask_dict_qa = torch.load("./ckpt/QA_Squad_ConvQA_10epoch/ckpt-epoch10/subnet.pth")
            self.mask_dict_crs = torch.load("./ckpt/CRS_only_epoch20/ckpt-epoch20/subnet.pth")
            del self.mask_dict_tod["shared.weight"]
            del self.mask_dict_cc["shared.weight"]
            del self.mask_dict_qa["shared.weight"]
            del self.mask_dict_crs["shared.weight"]

            del self.mask_dict_tod['lm_head.weight']
            del self.mask_dict_cc['lm_head.weight']
            del self.mask_dict_qa['lm_head.weight']
            del self.mask_dict_crs['lm_head.weight']

            del self.mask_dict_tod['resp_lm_head.weight']
            del self.mask_dict_cc['resp_lm_head.weight']
            del self.mask_dict_qa['resp_lm_head.weight']
            del self.mask_dict_crs['resp_lm_head.weight']

        train_dataset = MultiWOZDataset(self.reader, 'train', self.cfg.task, self.cfg.ururu,
                                        context_size=self.cfg.context_size, num_dialogs=self.cfg.num_train_dialogs,
                                        excluded_domains=self.cfg.excluded_domains)

        if self.cfg.num_gpus > 1:
            train_sampler = DistributedSampler(train_dataset)
        else:
            train_sampler = sampler.RandomSampler(train_dataset)

        train_collator = CollatorTrain(self.reader.pad_token_id, self.reader.tokenizer)

        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.cfg.batch_size_per_gpu,collate_fn=train_collator)
        num_training_steps_per_epoch = len(train_dataloader)
        optimizer, scheduler = self.get_optimizer_and_scheduler(
            num_training_steps_per_epoch, self.cfg.batch_size_per_gpu)

        reporter = Reporter(self.cfg.log_frequency, self.cfg.model_dir)

        max_score = 0
        for epoch in range(1, self.cfg.epochs + 1):
            if self.cfg.num_gpus > 1:
                train_dataloader.sampler.set_epoch(epoch)

            train_loss = self.train_epoch(train_dataloader, optimizer, scheduler, reporter)

            if self.cfg.num_gpus > 1:
                torch.distributed.barrier()

            logger.info("done {}/{} epoch, train loss is:{:f}".format(epoch, self.cfg.epochs, train_loss))


            if self.cfg.save_best_model:
                if self.cfg.local_rank == 0:
                    current_score = self.predict()
                    if max_score < current_score:
                        max_score = current_score
                        self.save_model(epoch)
            else:
                if self.cfg.local_rank in [0, -1]:
                    self.save_model(epoch)
            '''
            if self.cfg.local_rank in [0, -1]:
                self.save_model(epoch)
            '''
            if self.cfg.num_gpus > 1:
                torch.distributed.barrier()

    def _train_epoch(self, train_iterator, optimizer, scheduler, reporter=None):
        self.model.train()
        self.model.zero_grad()

        for step, batch in enumerate(train_iterator):
            start_time = time.time()

            inputs, labels = batch

            _, belief_labels, _ = labels

            loss, step_outputs = self.step_fn(inputs, *labels)

            if self.cfg.grad_accum_steps > 1:
                loss = loss / self.cfg.grad_accum_steps

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_grad_norm)

            if (step + 1) % self.cfg.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                lr = scheduler.get_last_lr()[0]

                if reporter is not None:
                    reporter.step(start_time, lr, step_outputs)

    def _train(self):
        train_batches, num_training_steps_per_epoch, _, _ = self.iterator.get_batches(
            "train", self.cfg.batch_size, self.cfg.num_gpus, shuffle=True,
            num_dialogs=self.cfg.num_train_dialogs, excluded_domains=self.cfg.excluded_domains)

        optimizer, scheduler = self.get_optimizer_and_scheduler(
            num_training_steps_per_epoch, self.cfg.batch_size)

        reporter = Reporter(self.cfg.log_frequency, self.cfg.model_dir)

        for epoch in range(1, self.cfg.epochs + 1):
            train_iterator = self.iterator.get_data_iterator(
                train_batches, self.cfg.task, self.cfg.ururu, self.cfg.add_auxiliary_task, self.cfg.context_size)

            self.train_epoch(train_iterator, optimizer, scheduler, reporter)

            logger.info("done {}/{} epoch".format(epoch, self.cfg.epochs))

            self.save_model(epoch)

            if not self.cfg.no_validation:
                self.validation(reporter.global_step)

    def _validation(self, global_step):
        self.model.eval()

        dev_batches, num_steps, _, _ = self.iterator.get_batches(
            "dev", self.cfg.batch_size, self.cfg.num_gpus)

        dev_iterator = self.iterator.get_data_iterator(
            dev_batches, self.cfg.task, self.cfg.ururu, self.cfg.add_auxiliary_task, self.cfg.context_size)

        reporter = Reporter(1000000, self.cfg.model_dir)

        torch.set_grad_enabled(False)
        for batch in tqdm(dev_iterator, total=num_steps, desc="Validaction"):
            start_time = time.time()

            inputs, labels = batch

            _, step_outputs = self.step_fn(inputs, *labels)

            reporter.step(start_time, lr=None, step_outputs=step_outputs, is_train=False)

        do_span_stats = True if "span" in step_outputs else False
        do_resp_stats = True if "resp" in step_outputs else False

        reporter.info_stats("dev", global_step, do_span_stats, do_resp_stats)

        torch.set_grad_enabled(True)

    def search_subnet(self):
        with open(os.path.join(self.cfg.ckpt, 'pytorch_model.bin'), "rb") as f:
            state = torch.load(
                f,
                map_location=(
                    lambda s, _: torch.serialization.default_restore_location(s, "cpu")
                ),
            )
        mask_dict = gen_mask_with_prob(state, self.cfg.mask_prob, gen_part='all', random_gen=self.cfg.gen_random_mask,
                                       include_embedding=self.cfg.include_embedding,
                                       exclude_output_proj=self.cfg.exclude_output_proj)

        mask_save_path = os.path.join(self.cfg.ckpt, 'subnet.pth')
        torch.save(mask_dict, mask_save_path)

        return mask_dict

    def finalize_bspn(self, belief_outputs, domain_history, constraint_history, span_outputs=None, input_ids=None):
        eos_token_id = self.reader.get_token_id(definitions.EOS_BELIEF_TOKEN)

        batch_decoded = []
        for i, belief_output in enumerate(belief_outputs):
            if belief_output[0] == self.reader.pad_token_id:
                belief_output = belief_output[1:]

            if eos_token_id not in belief_output:
                eos_idx = len(belief_output) - 1
            else:
                eos_idx = belief_output.index(eos_token_id)

            bspn = belief_output[:eos_idx + 1]

            decoded = {}

            decoded["bspn_gen"] = bspn
            #todo see yixia
            # update bspn using span output
            if span_outputs is not None and input_ids is not None:
                span_output = span_outputs[i]
                input_id = input_ids[i]

                eos_idx = input_id.index(self.reader.eos_token_id)
                input_id = input_id[:eos_idx]

                span_result = {}

                bos_user_id = self.reader.get_token_id(definitions.BOS_USER_TOKEN)

                span_output = span_output[:eos_idx]

                b_slot = None
                for t, span_token_idx in enumerate(span_output):
                    turn_id = max(input_id[:t].count(bos_user_id) - 1, 0)
                    turn_domain = domain_history[i][turn_id]

                    if turn_domain not in definitions.INFORMABLE_SLOTS:
                        continue

                    span_token = self.reader.span_tokens[span_token_idx]

                    if span_token not in definitions.INFORMABLE_SLOTS[turn_domain]:
                        b_slot = span_token
                        continue

                    if turn_domain not in span_result:
                        span_result[turn_domain] = defaultdict(list)

                    if b_slot != span_token:
                        span_result[turn_domain][span_token] = [input_id[t]]
                    else:
                        span_result[turn_domain][span_token].append(input_id[t])

                    b_slot = span_token

                for domain, sv_dict in span_result.items():
                    for s, v_list in sv_dict.items():
                        value = v_list[-1]
                        span_result[domain][s] = self.reader.tokenizer.decode(
                            value, clean_up_tokenization_spaces=False)

                span_dict = copy.deepcopy(span_result)

                ontology = self.reader.db.extractive_ontology

                flatten_span = []
                for domain, sv_dict in span_result.items():
                    flatten_span.append("[" + domain + "]")

                    for s, v in sv_dict.items():
                        if domain in ontology and s in ontology[domain]:
                            if v not in ontology[domain][s]:
                                del span_dict[domain][s]
                                continue

                        if s == "destination" or s == "departure":
                            _s = "destination" if s == "departure" else "departure"

                            if _s in sv_dict and v == sv_dict[_s]:
                                if s in span_dict[domain]:
                                    del span_dict[domain][s]
                                if _s in span_dict[domain]:
                                    del span_dict[domain][_s]
                                continue

                        if s in ["time", "leave", "arrive"]:
                            v = v.replace(".", ":")
                            if re.match("[0-9]+:[0-9]+", v) is None:
                                del span_dict[domain][s]
                                continue
                            else:
                                span_dict[domain][s] = v

                        flatten_span.append("[value_" + s + "]")
                        flatten_span.append(v)

                    if len(span_dict[domain]) == 0:
                        del span_dict[domain]
                        flatten_span.pop()


                decoded["span"] = flatten_span

                constraint_dict = self.reader.bspn_to_constraint_dict(
                    self.reader.tokenizer.decode(bspn, clean_up_tokenization_spaces=False))

                if self.cfg.overwrite_with_span:
                    _constraint_dict = OrderedDict()

                    for domain, slots in definitions.INFORMABLE_SLOTS.items():
                        if domain in constraint_dict or domain in span_dict:
                            _constraint_dict[domain] = OrderedDict()

                        for slot in slots:
                            if domain in constraint_dict:
                                cons_value = constraint_dict[domain].get(slot, None)
                            else:
                                cons_value = None

                            if domain in span_dict:
                                span_value = span_dict[domain].get(slot, None)
                            else:
                                span_value = None

                            if cons_value is None and span_value is None:
                                continue

                            # priority: span_value > cons_value
                            slot_value = span_value or cons_value

                            _constraint_dict[domain][slot] = slot_value
                else:
                    _constraint_dict = copy.deepcopy(constraint_dict)

                bspn_gen_with_span = self.reader.constraint_dict_to_bspn(
                    _constraint_dict)

                bspn_gen_with_span = self.reader.encode_text(
                    bspn_gen_with_span,
                    bos_token=definitions.BOS_BELIEF_TOKEN,
                    eos_token=definitions.EOS_BELIEF_TOKEN)

                decoded["bspn_gen_with_span"] = bspn_gen_with_span

            batch_decoded.append(decoded)

        return batch_decoded

    def finalize_resp(self, resp_outputs):
        bos_action_token_id = self.reader.get_token_id(definitions.BOS_ACTION_TOKEN)
        eos_action_token_id = self.reader.get_token_id(definitions.EOS_ACTION_TOKEN)

        bos_resp_token_id = self.reader.get_token_id(definitions.BOS_RESP_TOKEN)
        eos_resp_token_id = self.reader.get_token_id(definitions.EOS_RESP_TOKEN)

        batch_decoded = []
        for resp_output in resp_outputs:
            resp_output = resp_output[1:]
            if self.reader.eos_token_id in resp_output:
                eos_idx = resp_output.index(self.reader.eos_token_id)
                resp_output = resp_output[:eos_idx]

            try:
                bos_action_idx = resp_output.index(bos_action_token_id)
                eos_action_idx = resp_output.index(eos_action_token_id)
            except ValueError:
                #logger.warn("bos/eos action token not in : {}".format(
                #    self.reader.tokenizer.decode(resp_output)))
                aspn = [bos_action_token_id, eos_action_token_id]
            else:
                aspn = resp_output[bos_action_idx:eos_action_idx + 1]

            try:
                bos_resp_idx = resp_output.index(bos_resp_token_id)
                eos_resp_idx = resp_output.index(eos_resp_token_id)
            except ValueError:
                #logger.warn("bos/eos resp token not in : {}".format(
                #    self.reader.tokenizer.decode(resp_output)))
                resp = [bos_resp_token_id, eos_resp_token_id]
            else:
                resp = resp_output[bos_resp_idx:eos_resp_idx + 1]

            decoded = {"aspn_gen": aspn, "resp_gen": resp}

            batch_decoded.append(decoded)

        return batch_decoded

    def padding_context(self,contexts,pad=0):
        vectors=[]
        vec_lengths=[]
        #这里contexts是两层list  [['Hi', 'I', 'am', 'looking', 'for', 'a', 'movie', 'like', '@111776']]
        contexts_com=[]
        for sen in contexts[-5:-1]:
            contexts_com.extend(sen)
            contexts_com.append('_split_')
        contexts_com.extend(contexts[-1])
        vec,v_l,concept_mask,dbpedia_mask=self.padding_w2v(contexts_com,256,True)#concept_mask对应Key2index_3rd   dbpedia_mask对应电影的id  vec对应word2index_redial
        return vec,v_l,concept_mask,dbpedia_mask,0

    def padding_w2v(self,sentence,max_length,transformer=True,pad=0,end=2,unk=3):
        vector=[]
        concept_mask=[]
        dbpedia_mask=[]
        c=-1
        for i,word in enumerate(sentence):
            vector.append(self.word2index.get(word,unk))#词表
            #if word.lower() not in self.stopwords:
            concept_mask.append(self.key2index.get(word.lower(),0))
            #else:
            #    concept_mask.append(0)
            if word=="@":
                c=i+1
                dbpedia_mask.append(self.entity_max)
            elif i==c:
                try:
                    entity = self.id2entity[int(word)]
                    id=self.entity2entityId[entity]
                except:
                    id=self.entity_max
                dbpedia_mask.append(id)
            else:
                dbpedia_mask.append(self.entity_max)
        vector.append(end)
        concept_mask.append(0)
        dbpedia_mask.append(self.entity_max)

        if len(vector)>max_length:
            if transformer:
                return vector[-max_length:],max_length,concept_mask[-max_length:],dbpedia_mask[-max_length:]
            else:
                return vector[:max_length],max_length,concept_mask[:max_length],dbpedia_mask[:max_length]
        else:
            length=len(vector)
            return vector+(max_length-len(vector))*[pad],length,\
                   concept_mask+(max_length-len(vector))*[0],dbpedia_mask+(max_length-len(vector))*[self.entity_max]

    def bspn_to_db_recommend(self,user_history,turn):
         user_history[-1]=word_tokenize(user_history[-1])
         context_p=user_history
         response_p=word_tokenize(self.reader.tokenizer.decode(turn["resp"][1:-1]))
         context,c_lengths,concept_mask,dbpedia_mask,_=self.padding_context(context_p)
         assert len(context) == 256
         assert len(concept_mask) == 256
         assert len(dbpedia_mask) == 256
         bs=self.reader.tokenizer.decode(turn["bspn_gen"])
         entity=[]
         if "[value_entity]" in bs:
             start= bs.find("[value_entity]")+15
             entity_str=bs[start:-13]
             entity_p=entity_str.split(" ")
             for e in entity_p:
                 k="<http://dbpedia.org/resource/"+e+">"
                 try:
                     entity.append(self.e2id[k])
                 except:
                     warnings.warn("没找到"+k)
                     continue
         context=np.array(context)
         context=torch.from_numpy(context)
         context=torch.unsqueeze(context,0)


         c_lengths=torch.tensor([c_lengths])



         movie=torch.tensor([0])


         response=torch.from_numpy(np.zeros(30))
         response=torch.unsqueeze(response,0)

         mask_response=torch.from_numpy(np.zeros(30))
         mask_response=torch.unsqueeze(mask_response,0)

         entity_vec = np.zeros(64368)
         entity_vector = np.zeros(50, dtype=np.int)
         point = 0
         for en in entity:
             entity_vec[en] = 1
             entity_vector[point] = en
             point += 1

         entity_vec=torch.from_numpy(entity_vec)
         entity_vec=torch.unsqueeze(entity_vec,0)

         entity_vector=torch.from_numpy(entity_vector)
         entity_vector=torch.unsqueeze(entity_vector,0)

         concept_vec = np.zeros(29309)
         for con in concept_mask:
             if con != 0:
                 concept_vec[con] = 1
         concept_vec=torch.from_numpy(concept_vec)
         concept_vec=torch.unsqueeze(concept_vec,0)

         rec=torch.tensor([1])

         db_vec = np.zeros(64368)
         for db in dbpedia_mask:
             if  db != 0:
                 db_vec[db] = 1
         db_vec = torch.from_numpy(db_vec)
         db_vec=torch.unsqueeze(db_vec,0)
         dbpedia_mask = np.array(dbpedia_mask)
         dbpedia_mask = torch.from_numpy(dbpedia_mask)
         dbpedia_mask=torch.unsqueeze(dbpedia_mask,0)

         concept_mask = np.array(concept_mask)
         concept_mask = torch.from_numpy(concept_mask)
         concept_mask = torch.unsqueeze(concept_mask, 0)

         return self.loop.top10(context,response,mask_response,concept_mask,dbpedia_mask,entity_vec,movie,concept_vec,db_vec,entity_vector,rec)

    def predict(self):
        self.model.eval()

        if self.cfg.num_gpus > 1:
            model = self.model.module
        else:
            model = self.model

        pred_batches, _, _, _ = self.iterator.get_batches(
            self.cfg.pred_data_type, self.cfg.batch_size_per_gpu_eval,
            1, excluded_domains=self.cfg.excluded_domains)
        early_stopping = True if self.cfg.beam_size > 1 else False
        eval_dial_list = None
        results = {}
        #对每个batch
        for dial_batch in tqdm(pred_batches, total=len(pred_batches), desc="Prediction"):
            batch_size = len(dial_batch)
            #每个item有这三个列表
            dial_history = [[] for _ in range(batch_size)]
            domain_history = [[] for _ in range(batch_size)]

            # 4/20 add user_history

            user_history=[[] for _ in range(batch_size)]

            constraint_dicts = [OrderedDict() for _ in range(batch_size)]
        #对同一turn的batch
            for turn_batch in self.iterator.transpose_batch(dial_batch):
                tihuan=[]
                batch_encoder_input_ids_1 = []
                batch_encoder_input_ids_2 = []
                #t:同1turn中的第几个item
                for t, turn in enumerate(turn_batch):

                    context, _ = self.iterator.flatten_dial_history(
                        dial_history[t], [], len(turn["user"]), self.cfg.context_size)
                    #4/20
                    if turn["turn_domain"][0]=="[recommend]":
                        user_history[t].append(self.reader.tokenizer.decode(turn["usdx"][1:-1]))

                    encoder_input_ids_1 = context + turn["user"] + [self.reader.eos_token_id]
                    batch_encoder_input_ids_1.append(self.iterator.tensorize(encoder_input_ids_1))


                    turn_domain = turn["turn_domain"][-1]

                    if "[" in turn_domain:
                        turn_domain = turn_domain[1:-1]

                    domain_history[t].append(turn_domain)

                batch_encoder_input_ids_1 = pad_sequence(batch_encoder_input_ids_1,
                                                       batch_first=True,
                                                       padding_value=self.reader.pad_token_id)
                batch_encoder_input_ids_1 = batch_encoder_input_ids_1.to(self.cfg.device)

                attention_mask = torch.where(
                    batch_encoder_input_ids_1 == self.reader.pad_token_id, 0, 1)
                # belief tracking
                #encoder
                with torch.no_grad():
                    encoder_outputs_1 = model(input_ids=batch_encoder_input_ids_1,
                                                 attention_mask=attention_mask,
                                                 return_dict=False,
                                                 encoder_only=True,
                                                 add_auxiliary_task=self.cfg.add_auxiliary_task)

                    span_outputs, encoder_hidden_states = encoder_outputs_1
                    if isinstance(encoder_hidden_states, tuple):
                        last_hidden_state = encoder_hidden_states[0]
                    else:
                        last_hidden_state = encoder_hidden_states
                    # wrap up encoder outputs
                    encoder_outputs_1 = BaseModelOutput(
                        last_hidden_state=last_hidden_state)
                    # 生成belief_state
                    belief_outputs = model.generate(encoder_outputs=encoder_outputs_1,
                                                         attention_mask=attention_mask,
                                                         eos_token_id=self.reader.eos_token_id,
                                                         max_length=200,
                                                         do_sample=self.cfg.do_sample,
                                                         num_beams=self.cfg.beam_size,
                                                         early_stopping=early_stopping,
                                                         temperature=self.cfg.temperature,
                                                         top_k=self.cfg.top_k,
                                                         top_p=self.cfg.top_p,
                                                         decoder_type="belief")

                belief_outputs = belief_outputs.cpu().numpy().tolist()
                if self.cfg.add_auxiliary_task:
                    pred_spans = span_outputs[1].cpu().numpy().tolist()

                    input_ids = batch_encoder_input_ids_1.cpu().numpy().tolist()
                else:
                    pred_spans = None
                    input_ids = None

                decoded_belief_outputs = self.finalize_bspn(
                    belief_outputs, domain_history, constraint_dicts, pred_spans, input_ids)

                for t, turn in enumerate(turn_batch):
                    turn.update(**decoded_belief_outputs[t])
                #DST FINISHED
                if self.cfg.task == "e2e":
                    dbpn = []
                    if self.cfg.use_true_dbpn:
                        for turn in turn_batch:
                            dbpn.append(turn["dbpn"])
                    else:
                        for i,turn in enumerate(turn_batch):

                            if self.cfg.add_auxiliary_task:
                                bspn_gen = turn["bspn_gen_with_span"]
                            else:
                                bspn_gen = turn["bspn_gen"]

                            #bspn_gen = turn["bspn"]

                            bspn_gen = self.reader.tokenizer.decode(
                                bspn_gen, clean_up_tokenization_spaces=False)

                            if turn["turn_domain"][0]!='[recommend]':
                                db_token = self.reader.bspn_to_db_pointer(bspn_gen,
                                                                      turn["turn_domain"])

                            #4/20
                            else:
                                if "recommend_chit" not in str(bspn_gen):
                                    top10 =self.bspn_to_db_recommend(user_history[i],turn)
                                    top10_str = ""
                                    for item in top10:
                                        top10_str += item + " "
                                    top10_str=top10_str.lower()
                                    db_token = "[db_recommend] " + top10_str
                                else:
                                    db_token = "[db_recommend] "
                            dbpn_gen = self.reader.encode_text(
                                db_token,
                                bos_token=definitions.BOS_DB_TOKEN,
                                eos_token=definitions.EOS_DB_TOKEN)
                            turn["dbpn_gen"] = dbpn_gen
                        for t, turn in enumerate(turn_batch):

                            context, _ = self.iterator.flatten_dial_history(
                                dial_history[t], [], (len(turn["user"])+len(turn["dbpn_gen"])+len(turn["bspn_gen"])), self.cfg.context_size)
                            # 4/20
                            encoder_input_ids_2 = context + turn["user"] + turn["bspn_gen"]+turn["dbpn_gen"]+ [self.reader.eos_token_id]
                            batch_encoder_input_ids_2.append(self.iterator.tensorize(encoder_input_ids_2))
                        batch_encoder_input_ids_2 = pad_sequence(batch_encoder_input_ids_2,
                                                                     batch_first=True,
                                                                     padding_value=self.reader.pad_token_id)
                        batch_encoder_input_ids_2 = batch_encoder_input_ids_2.to(self.cfg.device)

                        attention_mask_2 = torch.where(
                                batch_encoder_input_ids_2 == self.reader.pad_token_id, 0, 1)
                            #dbpn.append(dbpn_gen)
                    '''
                    for t, db in enumerate(dbpn):
                        if self.cfg.use_true_curr_aspn:
                            db += turn_batch[t]["aspn"]

                        # T5 use pad_token as start_decoder_token_id
                        dbpn[t] = [self.reader.pad_token_id] + db
                    '''

                    # aspn has different length
                    #如果用标注的aspn
                    if self.cfg.use_true_curr_aspn:

                        for t, _dbpn in enumerate(dbpn):
                            resp_decoder_input_ids = self.iterator.tensorize([_dbpn])

                            resp_decoder_input_ids = resp_decoder_input_ids.to(self.cfg.device)

                            encoder_outputs = BaseModelOutput(
                                last_hidden_state=last_hidden_state[t].unsqueeze(0))

                            with torch.no_grad():
                                resp_outputs = model.generate(
                                    encoder_outputs=encoder_outputs,
                                    attention_mask=attention_mask[t].unsqueeze(0),
                                    decoder_input_ids=resp_decoder_input_ids,
                                    eos_token_id=self.reader.eos_token_id,
                                    max_length=300,
                                    do_sample=self.cfg.do_sample,
                                    num_beams=self.cfg.beam_size,
                                    early_stopping=early_stopping,
                                    temperature=self.cfg.temperature,
                                    top_k=self.cfg.top_k,
                                    top_p=self.cfg.top_p,
                                    decoder_type="resp")

                                resp_outputs = resp_outputs.cpu().numpy().tolist()

                                decoded_resp_outputs = self.finalize_resp(resp_outputs)

                                turn_batch[t].update(**decoded_resp_outputs[0])

                    else:
                        '''
                        dblist=[]
                        for i,d in enumerate(dbpn):
                            dblist.append(self.iterator.tensorize(d))
                        resp_decoder_input_ids=pad_sequence(dblist,batch_first=True,padding_value=self.reader.pad_token_id)
                        
                        #resp_decoder_input_ids = self.iterator.tensorize(dbpn)

                        resp_decoder_input_ids = resp_decoder_input_ids.to(self.cfg.device)
                        '''
                        with torch.no_grad():
                            encoder_outputs_2 = model(input_ids=batch_encoder_input_ids_2,
                                                           attention_mask=attention_mask_2,
                                                           return_dict=False,
                                                           encoder_only=True,
                                                           add_auxiliary_task=self.cfg.add_auxiliary_task)

                            span_outputs_2, encoder_hidden_states_2 = encoder_outputs_2
                            if isinstance(encoder_hidden_states_2, tuple):
                                last_hidden_state_2 = encoder_hidden_states_2[0]
                            else:
                                last_hidden_state_2 = encoder_hidden_states_2
                            # wrap up encoder outputs
                            encoder_outputs_2 = BaseModelOutput(
                                last_hidden_state=last_hidden_state_2)
                        # response generation
                        with torch.no_grad():
                            resp_outputs = model.generate(
                                encoder_outputs=encoder_outputs_2,
                                attention_mask=attention_mask_2,
                                eos_token_id=self.reader.eos_token_id,
                                max_length=300,
                                do_sample=self.cfg.do_sample,
                                num_beams=self.cfg.beam_size,
                                early_stopping=early_stopping,
                                temperature=self.cfg.temperature,
                                top_k=self.cfg.top_k,
                                top_p=self.cfg.top_p,
                                decoder_type="resp")

                        resp_outputs = resp_outputs.cpu().numpy().tolist()

                        decoded_resp_outputs = self.finalize_resp(resp_outputs)
                        for t, turn in enumerate(turn_batch):
                            #print(t)
                            '''
                            resp_tihuan=self.reader.resp_gen2resp_tihuan(decoded_resp_outputs[t]["resp_gen"],turn)
                            print(resp_tihuan)
                            tihuan.append(self.reader.encode_text(resp_tihuan))
                            '''

                            turn.update(**decoded_resp_outputs[t])
                            #print(str(t) + " " + self.reader.tokenizer.decode(turn["bspn_gen"]))
                            #print(str(t) + " " + self.reader.tokenizer.decode(turn["dbpn_gen"]))
                            #print(str(t) + " " + self.reader.tokenizer.decode(turn["aspn_gen"]))
                            #print(str(t)+ " "+self.reader.tokenizer.decode(decoded_resp_outputs[t]["resp_gen"]))

                            #print(tihuan)


                # update dial_history
                for t, turn in enumerate(turn_batch):
                    pv_text = copy.copy(turn["user"])
                    if turn["turn_domain"][0]=="[recommend]":

                        user_history[t].append(self.reader.tokenizer.decode(turn["resp_gen"][1:-1]))

                    if self.cfg.use_true_prev_bspn:
                        pv_bspn = turn["bspn"]
                    else:
                        if self.cfg.add_auxiliary_task:
                            pv_bspn = turn["bspn_gen_with_span"]
                        else:
                            pv_bspn = turn["bspn_gen"]

                    if self.cfg.use_true_dbpn:
                        pv_dbpn = turn["dbpn"]
                    else:
                        pv_dbpn = turn["dbpn_gen"]

                    if self.cfg.use_true_prev_aspn:
                        pv_aspn = turn["aspn"]
                    else:
                        pv_aspn = turn["aspn_gen"]

                    if self.cfg.use_true_prev_resp:
                        if self.cfg.task == "e2e":
                            pv_resp = turn["redx"]
                        else:
                            pv_resp = turn["resp"]
                    else:
                        pv_resp = turn["resp_gen"]
                    if self.cfg.ururu:
                        pv_text += pv_resp
                    else:
                        pv_text += (pv_bspn + pv_dbpn + pv_aspn + pv_resp)

                    dial_history[t].append(pv_text)

            result = self.iterator.get_readable_batch(dial_batch)
            results.update(**result)
            if self.cfg.data_type=="TOD":
                evaluator = MultiWozEvaluator(self.reader, self.cfg.pred_data_type)
                bleu, success, match = evaluator.e2e_eval(
                    results, eval_dial_list=eval_dial_list, add_auxiliary_task=self.cfg.add_auxiliary_task)
                score = 0.5 * (success + match) + bleu
                logger.info('match: %2.2f; success: %2.2f; bleu: %2.2f; score: %.2f' % (match, success, bleu, score))

        if self.cfg.output:
            save_json(results, os.path.join(self.cfg.ckpt, self.cfg.output))

        if self.cfg.data_type=="CC_UB" or self.cfg.data_type=="CC_FU" or self.cfg.data_type=="CC":
            evaluator = eval_chitchat.CC_evaluator(self.reader)
            bleu1,bleu2=evaluator.bleu(results)
            dict1,dict2=evaluator.dist(results)
            logger.info('bleu1: %.3f; bleu2: %.3f; dict1: %.3f; dict2: %.3f' % (bleu1, bleu2, dict1, dict2))
            score=bleu1+bleu2+dict1+dict2


        if self.cfg.data_type=="QA":
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
            f1=eval_QA.eval_qa("./data/CQA/coqa-dev-v1.0.json", "outqa.json")
            logger.info('f1: %.1f' % f1)
            score = f1


        if self.cfg.data_type=="QA_S":
            rqa = {}
            for dial_id, dial in results.items():
                for turn in dial:
                    turn_dict={}
                    if turn['turn_num']==0:
                        continue
                    rqa.update({turn["match"]:turn["resp_gen"][11:-11]})
                    save_json(rqa, "./outqa.json")

        if self.cfg.data_type=="CRS":
            recall=eval_CRS.recall(results)
            logger.info('recall: %.3f' % recall)
            score=recall

        return score




