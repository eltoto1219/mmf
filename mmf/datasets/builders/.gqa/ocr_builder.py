# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import Registry
from mmf.datasets.builders.vizwiz import VizWizBuilder
from mmf.datasets.builders.gqa.ocr_dataset import GQAOCRDataset


@Registry.register_builder("gqa_ocr")
class TextVQABuilder(VizWizBuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "GQA_OCR"
        self.set_dataset_class(GQAOCRDataset)

    @classmethod
    def config_path(self):
        return None
