# coding=utf-8
# Copyleft 2019 project LXRT.

from collections import defaultdict
from collections import namedtuple
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

Split2ImgFeatPath = {
    'mscoco_train': '/playpen2/lxmert_data/data/mscoco_imgfeat/train2014_obj36.tsv',
    'mscoco_minival': '/playpen2/lxmert_data/data/mscoco_imgfeat/val2014_obj36.tsv',
    'mscoco_nominival': '/playpen2/lxmert_data/data/mscoco_imgfeat/val2014_obj36.tsv',
    'vgnococo': '/playpen2/lxmert_data/data/vg_gqa_imgfeat/vg_gqa_obj36.tsv',
}

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

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
    print("Start to load Faster-RCNN detected objects from %s" % fname)
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
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
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
        collate_fn=BatchCollator(
            dataset_instance.dataset_name, dataset_instance.dataset_type),
        num_workers=num_workers,
        **other_args,
    )

    if num_workers >= 0:
        # Suppress leaking semaphore warning
        os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"
    loader.dataset_type = dataset_instance.dataset_type
    return loader, other_args.get("sampler", None)


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
        self._samplers = []
        for source in self.sources:
            self.data.extend(json.load(open("/playpen2/lxmert_data/data/lxmert/%s.json" % source)))
        print("Load %d data from %s" % (len(self.data), self.name))

        # Create answer table according to the qa_sets
        self.answer_table = AnswerTable(qa_sets)
        print("Load an answer table of size %d." % (len(self.answer_table.ans2id_map())))

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

    def __init__(self, dataset, dataset_name, dataset_type, topk=-1, config):

        super().__init__()
        self._dataset_name = dataset_name
        self._dataset_type = dataset_type
        self.raw_dataset = dataset
        self.task_matched = True
        self._add_answer = config.get("add_answer", True)
        self.config = config
        #self.masked_token_processor = MaskedTokenProcessor(config)

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

    def _add_masked_question(self, sample_info, current_sample):
        question = sample_info["sent"]
        if sample_info["label"] is None or sample["label"] == 0\
                or sample_info["is_matched"] != 1:
            # 1. No label 2. Label is pruned 3. unmatched visual + language pair
            ans = -1
        else:
            keys, values = zip(*example.label.items())
            if len(keys) == 1:
                ans = keys[0]
            else:
                value_sum = sum(values)
                prob = [value / value_sum for value in values]
                choice = np.random.multinomial(1, prob).argmax()
                ans = keys[choice]
        if ans == -1
            p_arg = {"text_a": question, "text_b": None, "is_correct": -1}
        else:
            p_arg = {"text_a": question, "text_b": ans, "is_correct": 1}

        processed = self.masked_token_processor(p_arg)
        processed.pop("tokens")
        current_sample.update(processed)

    def process_lxmert_datum(self, sample_info, sample):
        sample.image_id = object_to_byte_tensor(sample_info["image_id"])

        current_sample.img_info_0 = {
                "max_features": torch.tensor(36, dtype = torch.long),
                "bbox": torch.from_numpy(sample_info["boxes"]).cuda()
                "image_width": torch.tensor(sample_info['img_h'], dtype=torch.long ),
                "image_height": torch.tensor(sample_info['img_w'], dtype=torch.long),
                "cls_prob": torch.tensor(sample_info["obects_id"])}

        features = Sample({
            "image_feature_0":torch.from_numpy(sample_info["features"]).cuda()})
        current_sample.img0 = features

        #feature mask(15%)
        image_labels = []
        for i in range(features["image_feature_0"].shape[0]):
            prob = random.random()
            if prob < 0.15:
                if prob < 0.9:
                    features["image_feature_0"][i] = 0
                image_labels.append(1)
            else:
                # no masking token (will be ignored by loss function later)
                image_labels.append(-1)
        item = {}
        item["image_labels"] = image_labels
        sample.update(item)

        #object mask

        is_matched = 1
        if random.random() < 0.5:
            is_matched = 0
            other_datum = self.data[random.randint(0, len(self.data)-1)]
            while other_datum['img_id'] == img_id:
                other_datum = self.data[random.randint(0, len(self.data)-1)]
            sample["sent"] = other_datum['sent']
        sample.is_matched = torch.tensor(is_matched, dtype=torch.long)

        #Hao multiplies by the conifidence and also uses attribute_ids

        # answers will be included here
        if 'label' not in sample_info:
            sample_info["label"] = 0
        sample = self._add_masked_question(sample_info, current_sample)

        # extras
        if not self.use_ocr:
            # remove all OCRs from the sample
            # (i.e. make an empty OCR list)
            sample_info["ocr_tokens"] = []
            sample_info["ocr_info"] = []
            if "ocr_normalized_boxes" in sample_info:
                sample_info["ocr_normalized_boxes"] = np.zeros((0, 4), np.float32)
            # clear OCR visual features
            #if "image_feature_1" in sample:
            #    sample.image_feature_1 = torch.zeros_like(sample.image_feature_1)
            #return sample

        # Get PHOC embeddings for OCR tokens
        if hasattr(self, "phoc_processor"):
            context_phoc = self.phoc_processor({"tokens": ocr_tokens})
            sample.context_feature_1 = context_phoc["text"]
            sample.context_info_1 = Sample()
            sample.context_info_1.max_features = context_phoc["length"]

        # OCR order vectors
        if self.config.get("use_order_vectors", False):
            order_vectors = np.eye(len(sample.ocr_tokens), dtype=np.float32)
            order_vectors = torch.from_numpy(order_vectors)
            order_vectors[context["length"] :] = 0
            sample.order_vectors = order_vectors

         elif self.use_ocr_info and "ocr_info" in sample_info:
            # Old imdb format: OCR bounding boxes are computed on-the-fly
            # from ocr_info
            sample.ocr_bbox_coordinates = self.bbox_processor(
                {"info": sample_info["ocr_info"]}
            )["bbox"].coordinates

        current_sample = self._add_masked_question(sample_info, current_sample)

        return sample

    def __getitem__(self, item: int):
        current_sample = Sample()
        datum = self.data[item]
        img_info = self.imgid2img[img_id]
        datum['num_boxes'] = img_info['num_boxes']
        datum['features'] = img_info['features'].copy()
        datum['boxes'] = img_info['boxes'].copy()
        datum['objects_id'] = img_info['objects_id'].copy()
        datum['objects_conf'] = img_info['objects_conf'].copy()
        datum['attrs_id'] = img_info['attrs_id'].copy()
        datum['attrs_conf'] = img_info['attrs_conf'].copy()
        current_sample.uid = torch.tensor(
            datum["uid"], dtype=torch.int
        )
        datum.pop("uid")
        current_sample = self.process_lxmert_datum(datum, current_sample)
        return sample

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
        num_wokers=True):

        self._dataset_type = dataset_type
        self.dataset_name=dataset_name
        self.writer = registry.get("writer")
        self._global_config = registry.get("config")
        self._is_master = is_master()

        if qa_sets is not None:
            self.qa_sets = set(qa_set.lower().strip() for qa_set in qa_sets.split(",")

        self.topk=topk
        self.pin_memory = pin_memory
        self.num_workers=num_workers
        self.shuffle=shuffle
        self.drop_last=drop_last

    def _process_datasets(self):
        if "datasets" not in self.config:
            self.writer.write(
                "No datasets attribute present. Setting default to vqa2." "warning"
            )
            datasets = "vqa2"
        else:
            datasets = self.config.datasets

        if type(datasets) == str:
            datasets = list(map(lambda x: x.strip(), datasets.split(",")))

        self._given_datasets = dataset

    def load(self, config)
        self.config = config
        self._datasets = []
        self._loaders = []
        self._samplers = []
        self._iterators = []

        self._total_length = 0
        self._per_dataset_lengths = []
        self._num_datasets = 0
        self._finished_iterators = {}
        self._used_once = {}

        dset = LXMERTDataset(self.splits, qa_sets=self.qa_sets)
        dataset_instance = LXMERTTorchDataset(dataset: dset,
            dataset_name,
            dataset_type,
            topk=-1, config)

        loader_intance, sampler_instance = build_dataloader_and_sampler(dataset_instance,
            num_wokers,pin_memory,bs,shuffle, drop_last)

        self._datasets.append(dataset_instance)
        self._loaders.append(loader_instance)
        self._samplers.append(sampler_instance)

        self._num_datasets = 1
        sefl._dataset_probablities = 1
        training = self._global_config.training

        if self._dataset_type != "train":
            self._proportional_sampling = True

        if self._proportional_sampling is True:
            self._dataset_probablities =  1

        self._loader_index = 0
        self._chosen_dataset = self._datasets[self._loader_index]
        self._chosen_loader = self._loaders[self._loader_index]

    def __iter__(self):
        if self._num_datasets == 1:
            return iter(self._loaders[0])

        self._iterators = []
        self._finished_iterators = {}
        self._used_once = {}

        for loader in self._loaders:
            self._iterators.append(iter(loader))

        self._chosen_iterator = self._iterators[self._loader_index]

        return self

    def __next__(self):
        try:
            next_batch = next(self._chosen_iterator)
        except StopIteration:
            if (
                self._proportional_sampling is True
                or len(self._used_once) != self._num_datasets
            ):
                self._finished_iterators[self._loader_index] = 1

                if len(self._finished_iterators) == self._num_datasets:
                    raise
                else:
                    self.change_dataloader()
                next_batch = next(self._chosen_iterator)
            else:
                raise

        self._used_once[self._loader_index] = 1
        return next_batch

    def change_dataloader(self):
        if self._num_datasets <= 1:
            return
        choice = 0

        if self._is_master:
            choice = np.random.choice(
                self._num_datasets, 1, p=self._dataset_probablities
            )[0]

            while choice in self._finished_iterators:
                choice = np.random.choice(
                    self._num_datasets, 1, p=self._dataset_probablities
                )[0]

        choice = broadcast_scalar(choice, 0, device=registry.get("current_device"))
        self._loader_index = choice
        self._chosen_dataset = self._datasets[self._loader_index]
        self._chosen_loader = self._loaders[self._loader_index]
        self._chosen_iterator = self._iterators[self._loader_index]

    def verbose_dump(self, *args, **kwargs):
        self._chosen_dataset.verbose_dump(*args, **kwargs)

    def prepare_batch(self, batch):
        batch = self._chosen_dataset.prepare_batch(batch)
        self.change_dataloader()
        return batch

    def update_registry_for_model(self, config):
        """
        Use this if there is some specific configuration required by model
        which must be inferred at runtime.
        """
        for builder in self._builders:
            builder.update_registry_for_model(config)

    def clean_config(self, config):
        """
        Override this in case you want to clean the config you updated earlier
        in update_registry_for_model
        """
        return config

    def seed_sampler(self, epoch):
        if torch.distributed.is_initialized():
            for sampler in self._samplers:
                assert hasattr(
                    sampler, "set_epoch"
                ), "Can't seed without `set_epoch` method"
                sampler.set_epoch(epoch)

    @property
    def dataset_type(self):
        return self._dataset_type

    @property
    def current_dataset_name(self):
        return self._chosen_dataset.name

    @property
    def num_datasets(self):
        return self._num_datasets

    def get_datasets(self):
        return self._datasets

    @property
    def first_loader(self):
        return self._loaders[0]
