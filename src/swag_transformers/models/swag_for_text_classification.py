from argparse import ArgumentParser
import logging
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn.functional import softmax
from torch.optim import Adam, SGD
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm
from transformers import AdamW

import transformers

from swag.posteriors.swag import SWAG


logger = logging.getLogger(__name__)


class SwagForTextClassification(SWAG):
    def __init__(
        self, 
        base_model_class=BertForSequenceClassification, 
        base_model_subtype='bert-base-cased' 
        base_tokenizer_class=BertTokenizer, 
        base_tokenizer_subtype='bert-base-cased',
        n_labels=2, 
        no_cov_mat=True, 
        max_num_models=0, 
        var_clamp=1e-30, 
        device='cpu', 
        *args, 
        **kwargs
    ):
        super().__init__(base, no_cov_mat=no_cov_mat, max_num_models= max_num_models, var_clamp=var_clamp, device=device, *args, **kwargs)
        
        self.base_model = base_model_cls.from_pretrained(base_model_subtype, num_labels=num_labels) 
        self.tokenizer = base_tokenizer_cls.from_pretrained(base_tokenizer_subtype)




