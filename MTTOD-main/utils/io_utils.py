"""
   MTTOD: utils/io_utils.py

   implements simple I/O utilities for serialized objects and
   logger definitions.

   Copyright 2021 ETRI LIRS, Yohan Lee. All rights reserved.

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
import json
import pickle
import logging
import torch


def save_json(obj, save_path, indent=4):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def load_json(load_path, lower=True):
    with open(load_path, "r", encoding="utf-8") as f:
        obj = f.read()

        if lower:
            obj = obj.lower()

        return json.loads(obj)


def save_pickle(obj, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(load_path):
    with open(load_path, "rb") as f:
        return pickle.load(f)


def save_text(obj, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        for o in obj:
            f.write(o + "\n")


def load_text(load_path, lower=True):
    with open(load_path, "r", encoding="utf-8") as f:
        text = f.read()
        if lower:
            text = text.lower()
        return text.splitlines()


def get_or_create_logger(logger_name=None, log_dir=None):
    logger = logging.getLogger(logger_name)

    # check whether handler exists
    if len(logger.handlers) > 0:
        return logger

    # set default logging level
    logger.setLevel(logging.DEBUG)

    # define formatters
    stream_formatter = logging.Formatter(
        fmt="%(asctime)s  [%(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S")

    file_formatter = logging.Formatter(
        fmt="%(asctime)s  [%(levelname)s] %(module)s; %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S")

    # define and add handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, "log"))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

def gen_mask_with_prob(state, p, gen_part="all", random_gen=False, include_embedding=False,exclude_output_proj=False):
    """
    generate mask with probability
    :return: mask_dict
    """

    def gen_each_random_mask_with_prob(weight, p):
        """
        :param weight: a tensor
        :param p: probability
        :return:
        """
        mask = torch.rand_like(weight.float()) > p

        return mask

    def gen_each_mask_with_prob(weight, p):
        """
        :param weight: a tensor
        :param p: probability
        :return:
        """
        total_size = weight.nelement()
        kth = int(p * total_size)
        weight = torch.abs(weight.float())
        kth_element, _ = torch.kthvalue(weight.view(-1), k=kth, dim=0)
        kth_element = kth_element.tolist()  # float
        mask = weight > kth_element

        return mask

    def gen_embedding_mask_with_prob(weight, p):
        # mask embedding is a little different, we treat each vector in embedding weight as a linear.
        weight = torch.abs(weight.float())
        dim = weight.size(1)
        kth = int(p * dim)
        kth_element, _ = torch.kthvalue(weight, k=kth, dim=1, keepdim=True)  # kth_element: (num_vec,1)
        mask = weight > kth_element

        return mask


    mask_dict = {}
    gen_func = gen_each_random_mask_with_prob if random_gen else gen_each_mask_with_prob
    for k, v in state.items():
        if "weight" in k and "embed" not in k.lower() and "layer_norm" not in k:
            if gen_part == "all":
                mask_dict[k] = gen_func(v, p)
            elif gen_part == "encoder":
                if "encoder" in k:
                    mask_dict[k] = gen_func(v, p)
            elif gen_part == "decoder":
                if "decoder" in k:
                    mask_dict[k] = gen_func(v, p)
            else:
                raise NotImplementedError
        if include_embedding and "weight" in k and "embed" in k.lower():
            mask_dict[k] = gen_embedding_mask_with_prob(v, p)

    if exclude_output_proj:
        for k in list(state['model'].keys()):
            if 'output_projection' in k:
                print("Delete output projection")
                del state['model'][k]
    return mask_dict
