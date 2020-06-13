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


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class LXRTPretraining(BertPreTrainedModel):
    def __init__(self,
                 config,
                 task_mask_lm=True,
                 task_matched=True,
                 task_obj_predict=True,
                 visual_losses='',
                 task_qa=True,
                 num_answers=2):
        super().__init__(config)
        # Configuration
        self.config = config
        self.num_answers = num_answers

        # Use of pre-training tasks
        self.task_mask_lm = task_mask_lm
        self.task_obj_predict = task_obj_predict
        self.task_matched = task_matched
        self.task_qa = task_qa

        # LXRT backbone
        self.bert = LXRTModel(config)

        # Pre-training heads
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        if self.task_obj_predict:
            self.obj_predict_head = BertVisualObjHead(config, visual_losses)
        if self.task_qa:
            self.answer_head = BertVisualAnswerHead(config, self.num_answers)

        # Weight initialization
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                visual_feats=None, pos=None, obj_labels=None, matched_label=None, ans=None):
        (lang_output, visn_output), pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask,
            visual_feats=(visual_feats, pos),
        )

        lang_prediction_scores, cross_relationship_score = self.cls(lang_output, pooled_output)
        if self.task_qa:
            answer_score = self.answer_head(pooled_output)
        else:
            # This answer_score would not be used anywhere,
            # just to keep a constant return function signature.
            answer_score = pooled_output[0][0]


        loss_fct = CrossEntropyLoss(ignore_index=-1)
        losses = ()
        if masked_lm_labels is not None and self.task_mask_lm:
            masked_lm_loss = loss_fct(
                lang_prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1)
            )
            total_loss += masked_lm_loss
            output['masked_lm_loss'] = masked_lm_loss
        if matched_label is not None and self.task_matched:
            matched_loss = loss_fct(
                cross_relationship_score.view(-1, 2),
                matched_label.view(-1)
            )
            total_loss += matched_loss
            output['matched_loss'] = matched_loss
        if obj_labels is not None and self.task_obj_predict:
            loss_fcts = {
                'l2': SmoothL1Loss(reduction='none'),
                'ce': CrossEntropyLoss(ignore_index=-1, reduction='none')
            }
            visn_prediction_scores_dict = self.obj_predict_head(visn_output)
            for key in VISUAL_CONFIG.visual_losses:
                label, mask_conf = obj_labels[key]
                output_dim, loss_fct_name, label_shape, weight = VISUAL_CONFIG.visual_loss_config[key]
                visn_loss_fct = loss_fcts[loss_fct_name]
                visn_prediction_scores = visn_prediction_scores_dict[key]
                visn_loss = visn_loss_fct(
                    visn_prediction_scores.view(-1, output_dim),
                    label.view(*label_shape),
                )
                if visn_loss.dim() > 1:     # Regression Losses
                    visn_loss = visn_loss.mean(1)
                visn_loss = (visn_loss * mask_conf.view(-1)).mean() * weight
                output['visn_loss'] = visn_loss
            
        if ans is not None and self.task_qa:
            answer_loss = loss_fct(
                answer_score.view(-1, self.num_answers),
                ans.view(-1)
            )  
            # Since this Github version pre-trains with QA loss from the beginning,
            # I exclude "*2" here to match the effect of QA losses.
            # Previous: (loss *0) for 6 epochs, (loss *2) for 6 epochs.   (Used 10 instead of 6 in EMNLP paper)
            # Now     : (loss *1) for 12 epochs
            #
            # * 2       # Multiply by 2 because > half of the data will not have label
            output['answer_loss'] = answer_loss
        return output
