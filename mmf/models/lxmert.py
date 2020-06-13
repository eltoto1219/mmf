# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.utils.modeling import get_optimizer_parameters_for_bert

from mmf.models.lxmert_pretraining_model import LXRTPretraining
from mmf.models.lxmert_downstream_model import LXMERTForClassification


"""========================================"""
"""register model defined below"""
"""========================================"""



@registry.register_model("lxmert")
class LXMERT(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def config_path(cls):
        return "configs/models/LXMERT/pretrain.yaml"

    def build(self):
        if self.config.training_head_type == "pretraining":
            self.model = LXRTPretraining(self.config)
        else:
            self.model = LXMERTForClassification(self.config)

    def get_image_and_text_features(self, sample_list):
        """
        TO DO: This would be important
        """
        if self.config.training_head_type == "pretraining":
            #load all datasets
            pass
        else:
            # load respective datasets
            if sample_list.dataset_name == "nlvr2":
                pass
            elif sample_list.dataset_name == "gqa":
                pass
            elif sample_list.dataset_name == "vqa":
                pass
        return sample_list

    def get_optimizer_parameters(self, config):
        return get_optimizer_parameters_for_bert(self.model, config)

    def forward(self, sample_list):
        params = self.get_image_and_text_features(sample_list)        

        if self.config.training_head_type == "pretraining":
            output_dict = self.model(
                params["input_ids"],
                params["token_type_ids"],
                params["attention_mask"],
                params["masked_lm_labels"],
                params["visual_feats"],
                params["pos"],
                params["obj_labels"],
                params["matched_label"],
                params["ans"],
            )

            loss_key = "{}/{}".format(
                sample_list.dataset_name, sample_list.dataset_type
            )
            output_dict["losses"] = {}
            output_dict["losses"][loss_key + "/masked_lm_loss"] = output_dict.pop(
                "masked_lm_loss"
            )
            output_dict["losses"][loss_key + "/matched_loss"] = output_dict.pop(
                "matched_loss"
            )
            output_dict["losses"][loss_key + "/visn_loss"] = output_dict.pop(
                "visn_loss"
            )
            output_dict["losses"][loss_key + "/answer_loss"] = output_dict.pop(
                "answer_loss"
            )
        else:
            output_dict = self.model(
                params["input_ids"],
                params["token_type_ids"],
                params["attention_mask"],
                params["visual_feats"],
                params["pos"],
            ) #this output_dict contains only one thing: logits

        return output_dict
