# Copyright (c) Facebook, Inc. and its affiliates.

import math
import os
from copy import deepcopy
import copy
import json
import logging
import shutil
import tarfile
import tempfile
import sys
from io import open

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss

from mmf.models.lxmert_bert_utils import * #to do: juts import needed ones

"""========================================"""
"""downstream model defined below"""
"""========================================"""


class LXRTFeatureExtraction(BertPreTrainedModel):
    """
    BERT model for classification.
    """
    def __init__(self, config, mode='lxr'):
        """

        :param config:
        :param mode:  Number of visual layers
        """
        super().__init__(config)
        self.bert = LXRTModel(config)
        self.mode = mode
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, visual_feats=None,
                visual_attention_mask=None):
        feat_seq, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                            visual_feats=visual_feats,
                                            visual_attention_mask=visual_attention_mask)
        if 'x' == self.mode:
            return pooled_output
        elif 'x' in self.mode and ('l' in self.mode or 'r' in self.mode):
            return feat_seq, pooled_output
        elif 'l' in self.mode or 'r' in self.mode:
            return feat_seq


def set_visual_config(config):
    VISUAL_CONFIG.l_layers = config.llayers
    VISUAL_CONFIG.x_layers = config.xlayers
    VISUAL_CONFIG.r_layers = config.rlayers


class LXRTClassifierEncoder(nn.Module):
    def __init__(self, config, max_seq_length, mode='x'):
        super().__init__()
        self.max_seq_length = max_seq_length
        set_visual_config(config)

        # Build LXRT Model
        self.model = LXRTFeatureExtraction.from_pretrained(
            config.bert_model_name, #TO DO: this might cause bugs since we don'y know it can load successfully
            mode=mode
        )

        if config.from_scratch:
            print("initializing all the weights")
            self.model.apply(self.model.init_bert_weights)

    @property
    def dim(self):
        return 768

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, visual_feats=None, visual_attention_mask=None):
        output = self.model(input_ids, token_type_ids, attention_mask,
                            visual_feats=visual_feats,
                            visual_attention_mask=visual_attention_mask)
        return output

    def save(self, path):
        torch.save(self.model.state_dict(),
                   os.path.join("%s_LXRT.pth" % path))

    def load(self, path):
        # Load state_dict from snapshot file
        print("Load LXMERT pre-trained model from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        # Load weights to model
        self.model.load_state_dict(state_dict, strict=False)


class AnswerTable:
    ANS_CONVERT = {
        "a man": "man",
        "the man": "man",
        "a woman": "woman",
        "the woman": "woman",
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
        'grey': 'gray',
    }

    def __init__(self, dsets=None):
        self.all_ans = json.load(open("data/lxmert/all_ans.json"))
        if dsets is not None:
            dsets = set(dsets)
            # If the answer is used in the dsets
            self.anss = [ans['ans'] for ans in self.all_ans if
                         len(set(ans['dsets']) & dsets) > 0]
        else:
            self.anss = [ans['ans'] for ans in self.all_ans]
        self.ans_set = set(self.anss)

        self._id2ans_map = self.anss
        self._ans2id_map = {ans: ans_id for ans_id, ans in enumerate(self.anss)}

        assert len(self._id2ans_map) == len(self._ans2id_map)
        for ans_id, ans in enumerate(self._id2ans_map):
            assert self._ans2id_map[ans] == ans_id

    def convert_ans(self, ans):
        if len(ans) == 0:
            return ""
        ans = ans.lower()
        if ans[-1] == '.':
            ans = ans[:-1].strip()
        if ans.startswith("a "):
            ans = ans[2:].strip()
        if ans.startswith("an "):
            ans = ans[3:].strip()
        if ans.startswith("the "):
            ans = ans[4:].strip()
        if ans in self.ANS_CONVERT:
            ans = self.ANS_CONVERT[ans]
        return ans

    def ans2id(self, ans):
        return self._ans2id_map[ans]

    def id2ans(self, ans_id):
        return self._id2ans_map[ans_id]

    def ans2id_map(self):
        return self._ans2id_map.copy()

    def id2ans_map(self):
        return self._id2ans_map.copy()

    def used(self, ans):
        return ans in self.ans_set

    def all_answers(self):
        return self.anss.copy()

    @property
    def num_answers(self):
        return len(self.anss)


def load_lxmert_qa(path, model, label2ans):
    """
    Load model weights from LXMERT pre-training.
    The answers in the fine-tuned QA task (indicated by label2ans)
        would also be properly initialized with LXMERT pre-trained
        QA heads.

    :param path: Path to LXMERT snapshot.
    :param model: LXRT model instance.
    :param label2ans: The label2ans dict of fine-tuned QA datasets, like
        {0: 'cat', 1: 'dog', ...}
    :return:
    """
    print("Load QA pre-trained LXMERT from %s " % path)
    loaded_state_dict = torch.load("%s_LXRT.pth" % path)
    model_state_dict = model.state_dict()

    # Handle Multi-GPU pre-training --> Single GPU fine-tuning
    for key in list(loaded_state_dict.keys()):
        loaded_state_dict[key.replace("module.", '')] = loaded_state_dict.pop(key)

    # Isolate bert model
    bert_state_dict = {}
    for key, value in loaded_state_dict.items():
        if key.startswith('bert.'):
            bert_state_dict[key] = value

    # Isolate answer head
    answer_state_dict = {}
    for key, value in loaded_state_dict.items():
        if key.startswith("answer_head."):
            answer_state_dict[key.replace('answer_head.', '')] = value

    # Do surgery on answer state dict
    ans_weight = answer_state_dict['logit_fc.3.weight']
    ans_bias = answer_state_dict['logit_fc.3.bias']
    import copy
    new_answer_weight = copy.deepcopy(model_state_dict['logit_fc.3.weight'])
    new_answer_bias = copy.deepcopy(model_state_dict['logit_fc.3.bias'])
    answer_table = AnswerTable()
    loaded = 0
    unload = 0
    if type(label2ans) is list:
        label2ans = {label: ans for label, ans in enumerate(label2ans)}
    for label, ans in label2ans.items():
        new_ans = answer_table.convert_ans(ans)
        if answer_table.used(new_ans):
            ans_id_9500 = answer_table.ans2id(new_ans)
            new_answer_weight[label] = ans_weight[ans_id_9500]
            new_answer_bias[label] = ans_bias[ans_id_9500]
            loaded += 1
        else:
            new_answer_weight[label] = 0.
            new_answer_bias[label] = 0.
            unload += 1
    print("Loaded %d answers from LXRTQA pre-training and %d not" % (loaded, unload))
    print()
    answer_state_dict['logit_fc.3.weight'] = new_answer_weight
    answer_state_dict['logit_fc.3.bias'] = new_answer_bias

    # Load Bert Weights
    bert_model_keys = set(model.lxrt_encoder.model.state_dict().keys())
    bert_loaded_keys = set(bert_state_dict.keys())
    assert len(bert_model_keys - bert_loaded_keys) == 0
    model.lxrt_encoder.model.load_state_dict(bert_state_dict, strict=False)

    # Load Answer Logic FC Weights
    model_keys = set(model.state_dict().keys())
    ans_loaded_keys = set(answer_state_dict.keys())
    assert len(ans_loaded_keys - model_keys) == 0

    model.load_state_dict(answer_state_dict, strict=False)


class LXMERTForClassification(nn.Module):
    def __init__(self, config):

        # Model
        self.config = config       
        self.bert = LXRTClassifierEncoder(
            config,
            max_seq_length=config.MAX_SEQ_LENGTH
        )
        hid_dim = self.bert.dim

        self.training_head_type = self.config.training_head_type

        self.classifier = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.init_weights()

    def init_weights(self):
        #To Do: still unclear about initialization and where to put weights
        self.logit_fc.apply(self.bert.init_bert_weights)

        if self.config.load_lxmert is not None:
            self.bert.load(self.config.load_lxmert)
        if self.config.load_lxmert_qa is not None:
            load_lxmert_qa(self.config.load_lxmert_qa, self.bert,
                           label2ans=self.train_tuple.dataset.label2ans)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, visual_feats=None, pos=None):
        x = self.bert(input_ids, token_type_ids, attention_mask, (visual_feats, pos))
        logit = self.classifier(x)

        output = {}
        output["scores"] = logit

        return output

