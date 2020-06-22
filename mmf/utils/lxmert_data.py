# coding=utf-8
# Copyleft 2019 project LXRT.

from collections import defaultdict
from collections import namedtuple
import warnings
from collections import OrderedDict
from copy import deepcopy
import json
import random
import torch
import os
import csv
import base64
import numpy as np
from torch.utils.data import Dataset
import time
import sys
from mmf.datasets.mmf_dataset import MMFDataset
from mmf.utils.distributed import byte_tensor_to_object, object_to_byte_tensor
from mmf.utils.distributed import broadcast_scalar, is_master
from mmf.utils.general import get_batch_size
from mmf.common.registry import registry
from mmf.datasets.processors.bert_processors import MaskedTokenProcessor
from mmf.common.batch_collator import BatchCollator
from mmf.common.sample import SampleList, Sample

TINY_IMG_NUM = 500
FAST_IMG_NUM = 5000
SPLITS_VAL = "mscoco_minival"

Split2ImgFeatPath = {
    'mscoco_train': '/playpen2/lxmert_data/data/mscoco_imgfeat/train2014_obj36.tsv',
    'mscoco_minival': '/playpen2/lxmert_data/data/mscoco_imgfeat/val2014_obj36.tsv',
    'mscoco_nominival': '/playpen2/lxmert_data/data/mscoco_imgfeat/val2014_obj36.tsv',
    'vgnococo': '/playpen2/lxmert_data/data/vg_gqa_imgfeat/vg_gqa_obj36.tsv',
}

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

class Report2(OrderedDict):
    def __init__(self, batch_size, model_output=None, *args):
        super().__init__(self)
        if model_output is None:
            model_output = {}

        all_args = [model_output] + [*args]
        self.writer = registry.get("writer")
        self.batch_size = batch_size
        self.warning_string = (
            "Updating forward report with key {}"
            "{}, but it already exists in {}. "
            "Please consider using a different key, "
            "as this can cause issues during loss and "
            "metric calculations."
        )

        for idx, arg in enumerate(all_args):
            for key, item in arg.items():
                if key in self and idx >= 2:
                    log = self.warning_string.format(
                        key, "", "in previous arguments to report"
                    )
                    warnings.warn(log)

                if type(item) == torch.Tensor:
                    continue
                    #item = torch.cat((self[key], report[key]), dim=0)
                self[key] = item

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def fields(self):
        return list(self.keys())

class CustomBatch:
    def __init__(self, data):
        assert type(data) == list, "{}".format(data[0])
        for k, v in data[0].items():
            assert type(v) is torch.Tensor,"{}".format(v, k)
        keys = list(deepcopy(data[0]).keys())
        values = list(
                map(
                    torch.stack,
                    list(
                        zip(
                            *tuple(
                                map(
                                    lambda a: list(a.values())
                                    , data)
                                )
                            )
                        )
                    )
                )
        for k, v in zip(keys, values):
            assert type(v) == torch.Tensor
            setattr(self, k, v)

    def pin_memory(self):
        torch_attrs = list(filter(lambda x: "__" not in x and "pin_memory" not in x, dir(self)))
        for n in torch_attrs:
            a = getattr(self, n)
            setattr(self, n, a.pin_memory())
        return self

def collate_wrapper(batch):
    return CustomBatch(batch)

def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])

            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    return data


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
        self.all_ans = json.load(open("/playpen2/lxmert_data/data/lxmert/all_ans.json"))
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


def build_dataloader_and_sampler(
        dataset_instance,
        num_workers,
        pin_memory,
        batch_size,
        shuffle,
        drop_last):

    other_args = {"drop_last": drop_last}

    if torch.distributed.is_initialized():
        other_args["sampler"] = torch.utils.data.DistributedSampler(
            dataset_instance, shuffle=shuffle
        )
    else:
        other_args["shuffle"] = shuffle

    other_args["batch_size"] = batch_size

    loader = torch.utils.data.DataLoader(
        dataset=dataset_instance,
        pin_memory=pin_memory,
        collate_fn=collate_wrapper,
        #BatchCollator(
        #    dataset_instance.dataset_name, dataset_instance.dataset_type),
        num_workers=1,
        **other_args,
    )

    if num_workers >= 0:
        # Suppress leaking semaphore warning
        os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"
    return loader


class LXMERTDataset:
    def __init__(self, splits: str, qa_sets=None):
        """
        :param splits: The data sources to be loaded
        :param qa_sets: if None, no action
                        o.w., only takes the answers appearing in these dsets
                              and remove all unlabeled data (MSCOCO captions)
        """
        self.name = splits
        self.sources = splits.split(',')

        # Loading datasets to data
        self.data = []
        for source in self.sources:
            self.data.extend(json.load(open("/playpen2/lxmert_data/data/lxmert/%s.json" % source)))

        # Create answer table according to the qa_sets
        self.answer_table = AnswerTable(qa_sets)

        # Modify the answers
        for datum in self.data:
            labelf = datum['labelf']
            for cat, labels in labelf.items():
                for label in labels:
                    for ans in list(label.keys()):
                        new_ans = self.answer_table.convert_ans(ans)
                        if self.answer_table.used(new_ans):
                            if ans != new_ans:
                                label[new_ans] = label.pop(ans)
                        else:
                            label.pop(ans)

    def __len__(self):
        return len(self.data)

    def make_uid(self, img_id, dset, sent_idx):
        return "%s_%s_%03d" % (img_id, dset, sent_idx),

# NORMAL
class LXMERTTorchDataset(MMFDataset):

    def __init__(self, dataset, dataset_name, dataset_type, topk, config):
        #super().__init__(dataset_name, config)
        self._dataset_name = dataset_name
        self._dataset_type = dataset_type
        self.raw_dataset = dataset
        self.task_matched = True
        self._add_answer = config.get("add_answer", True)
        self.config = config
        #tokener congig

        CO =  namedtuple('config', "tokenizer_config max_seq_length")
        TC = namedtuple("tokenizer_config", "type use_lower_case params")
        c = CO(TC("bert-base-uncased", True, {}), 20)

        self.masked_token_processor = MaskedTokenProcessor(c)

        self.writer = registry.get("writer")
        self._global_config = registry.get("config")
        self._dataset_type = dataset_type
        self._dataset_name=dataset_name
        self._is_master = is_master()

        self._num_datasets = 1
        self._dataset_probablities = 1
        training = self._global_config.training
        self._dataset_probablities =  1

        if True:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM

        # Load the dataset
        img_data = []
        for source in self.raw_dataset.sources:
            img_data.extend(load_obj_tsv(Split2ImgFeatPath[source], topk))

        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Filter out the dataset
        used_data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                used_data.append(datum)

        # Flatten the dataset (into one sent + one image entries)
        self.data = []
        for datum in used_data:
            sentf = datum['sentf']
            for sents_cat, sents in sentf.items():
                if sents_cat in datum['labelf']:
                    labels = datum['labelf'][sents_cat]
                else:
                    labels = None
                for sent_idx, sent in enumerate(sents):
                    new_datum = {
                        'uid': self.make_uid(datum['img_id'], sents_cat, sent_idx),
                        'img_id': datum['img_id'],
                        'sent': sent
                    }
                    if labels is not None:
                        new_datum['label'] = labels[sent_idx]
                    self.data.append(new_datum)

    def make_uid(self, img_id, dset, sent_idx):
        return "%s_%s_%03d" % (img_id, dset, sent_idx)

    def _add_masked_question(self, sample_info, current_sample):
        question = sample_info["sent"]
        p_arg = {"text_a": question, "text_b": None, "is_correct": -1}
        processed = self.masked_token_processor(p_arg)
        processed.pop("tokens")
        current_sample = {**current_sample, **processed}
        return current_sample

    def process_lxmert_datum(self, sample_info, sample):
        #sample.image_id = torch.tensor(
        #    int(sample_info["img_id"]), dtype=torch.int
        #    )

        obj_labels = torch.from_numpy(sample_info["objects_id"])
        obj_conf = torch.from_numpy(sample_info["objects_conf"])
        #attr_labels = torch.from_numpy(sample_info["attrs_id"]).long()
        #attr_conf = torch.from_numpy(sample_info["attrs_conf"]).float()

        placeholder = torch.zeros(1, 36, 1600)
        for i in range(0, 36):
            placeholder[0][i][int(obj_labels[i])] = obj_conf[i]

        sample["cls_prob"] = placeholder
        sample["max_features"] = torch.tensor(36, dtype = torch.int)
        sample["image_width"] = torch.tensor(sample_info['img_w'], dtype=torch.int)
        sample["image_height"] = torch.tensor(sample_info['img_h'], dtype=torch.int)
        sample["bbox"] = torch.from_numpy(sample_info["boxes"])
        sample["image_feature_0"] = torch.from_numpy(sample_info["features"])
        sample["lxmert_custom"] = torch.tensor(1, dtype=torch.int)


        #sample.img_info_0 = {
        #        "max_features": torch.tensor(36, dtype = torch.int),
        #        "bbox": torch.from_numpy(sample_info["boxes"]),
        #        "image_width": torch.tensor(sample_info['img_h'], dtype=torch.int),
        #        "image_height": torch.tensor(sample_info['img_w'], dtype=torch.int),
        #        "cls_prob": placeholder}

        #features = Sample({
        #    "image_feature_0":torch.from_numpy(sample_info["features"])})
        #sample.img0 = features

        #feature mask(15%)
        image_labels = []
        for i in range(sample["image_feature_0"].shape[0]):
            prob = random.random()
            if prob < 0.15:
                if prob < 0.9:
                    sample["image_feature_0"][i] = 0
                image_labels.append(1)
            else:
                # no masking token (will be ignored by loss function later)
                image_labels.append(-1)

        sample["image_labels"] = torch.Tensor(image_labels)
        #sample.update(item)

        #object mask
        img_id = sample_info["img_id"]
        is_matched = 1
        if random.random() < 0.5:
            is_matched = 0
            other_datum = self.data[random.randint(0, len(self.data)-1)]
            while other_datum['img_id'] == img_id:
                other_datum = self.data[random.randint(0, len(self.data)-1)]
            sample_info["sent"] = other_datum['sent']
        #sample.is_matched = torch.tensor(is_matched, dtype=torch.int)
        sample["is_matched"] = torch.tensor(is_matched, dtype=torch.int)
        sample = self._add_masked_question(sample_info, sample)


        #answer
        if sample_info["label"] is None or sample_info["label"] == 0\
                or int(sample["is_matched"]) != 1\
                or len(sample_info["label"]) < 1:
            # 1. No label 2. Label is pruned 3. unmatched visual + language pair
            ans = -1
        else:
            assert len(sample_info["label"]) >= 1
            keys, values = zip(*iter(sample_info["label"].items()))
            #if len(keys) == 1:
            #    ans = keys[0]
            #else:
            #    value_sum = sum(values)
            #    prob = [value / value_sum for value in values]
            #    choice = np.random.multinomial(1, prob).argmax()
            #    ans = keys[choice]
            #keys = list(sample_info["label"].keys())
            #values = list(sample_info["label"].values())
            if len(keys) == 1:
                ans = keys[0]
            else:
                value_sum = sum(values)
                prob = [value / value_sum for value in values]
                choice = int(np.random.choice(
                    list(range(0,len(keys))), size = 1, p = prob))
                ans = keys[choice]

        sample["targets"] = torch.tensor(ans, dtype=torch.int)

        return sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):

        datum = self.data[item]
        img_id = datum["img_id"]

        if 'label' in datum:
            new_label = {}
            label = datum['label'].copy()
            for ans in list(label.keys()):
                new_label[self.raw_dataset.answer_table.ans2id(ans)] = label.pop(ans)
        else:
            new_label = None
        datum["label"] = new_label
        img_info = self.imgid2img[img_id]
        datum["img_w"] = img_info["img_w"]
        datum["img_h"] = img_info["img_h"]
        datum['num_boxes'] = img_info['num_boxes']
        datum['features'] = img_info['features'].copy()
        datum['boxes'] = img_info['boxes'].copy()
        datum['objects_id'] = img_info['objects_id'].copy()
        datum['objects_conf'] = img_info['objects_conf'].copy()
        datum['attrs_id'] = img_info['attrs_id'].copy()
        datum['attrs_conf'] = img_info['attrs_conf'].copy()

        current_sample = {}
        current_sample = self.process_lxmert_datum(datum, current_sample)

        return current_sample

# custom loader here
class LXMERTDatasetLoader:
    def __init__(self,
        config,
        splits: str,
        bs: int,
        shuffle=False,
        drop_last=False,
        topk=-1,
        qa_sets="vqa,gqa,visual7w",
        dataset_name="pretrain",
        dataset_type="train",
        pin_memory=True,
        num_workers=4):


        if qa_sets is not None:
            self.qa_sets = set(qa_set.lower().strip() for qa_set in qa_sets.split(","))
        self.topk=topk
        self.pin_memory = pin_memory
        self.num_workers=num_workers
        self.shuffle=shuffle
        self.drop_last=drop_last
        self.splits=splits
        self.bs=bs

        self._dataset_name = dataset_name
        self._dataset_type = dataset_type
        self.config = config

    def load_datasets(self):

        dset = LXMERTDataset(self.splits, qa_sets=self.qa_sets)
        dataset_instance = LXMERTTorchDataset(dset,
            self._dataset_name,
            self._dataset_type,
            self.topk,
            self.config)

        self.train_dataset = dataset_instance
        loader_instance = build_dataloader_and_sampler(dataset_instance,
            self.num_workers,self.pin_memory,self.bs,self.shuffle, self.drop_last)
        ### val instance
        vset = LXMERTDataset(SPLITS_VAL, qa_sets=self.qa_sets)
        val_instance = LXMERTTorchDataset(dset,
            "pretrain",
            "valid",
            5000,
            self.config)
        self.val_dataset = val_instance

        val_loader  = build_dataloader_and_sampler(dataset_instance,
            self.num_workers,self.pin_memory,self.bs,self.shuffle, self.drop_last)


        self.train_loader = loader_instance
        self.val_loader = val_loader

    @property
    def dataset_config(self):
        return self._dataset_config

    @dataset_config.setter
    def dataset_config(self, config):
        self._dataset_config = config

    def get_config(self):
        return self._dataset_config

    def get_test_reporter(self, dataset_type):
        dataset = getattr(self, f"{dataset_type}_dataset")
        return TestReporter(dataset)

    def prepare_batch(self, batch, *args, **kwargs):
        #batch = SampleList(batch)
        #return self.mapping[batch.dataset_type].prepare_batch(batch)
        return batch

    def verbose_dump(self, report, *args, **kwargs):
        if self.config.training.verbose_dump:
            dataset_type = report.dataset_type
            self.mapping[dataset_type].verbose_dump(report, *args, **kwargs)

    def seed_sampler(self, dataset_type, seed):
        pass
