
import logging

import torch

from transformers import logging as hf_logging


# Get the transformers logger
transformers_logger = hf_logging.get_logger()
# Remove existing handlers
transformers_logger.handlers = []
# Enable propagation to root logger
transformers_logger.propagate = True
hf_logging.set_verbosity_info()


def identical_models(model1, model2, log_same=False):
    """Compare two models and their parameters"""
    all_same = True
    weights_1 = {}
    weights_2 = {}
    for (module, name, full_name) in model1.swag.params:
        weights_1[full_name] = module.__getattr__(name)
    for (module, name, full_name) in model2.swag.params:
        weights_2[full_name] = module.__getattr__(name)
    for name in weights_1:
        same = torch.allclose(weights_1[name], weights_2[name])
        if not same:
            all_same = False
            logging.info("not close: %s", name)
        elif log_same:
            logging.info("close: %s", name)
    return all_same


def buf_and_param_names(model):
    """Return set of parameter and buffer names prefixed with BUF: or PAR:"""
    named_buffers = set('BUF:' + x[0] for x in model.named_buffers())
    named_params = set('PAR:' + x[0] for x in model.named_parameters())
    bufs_and_params = named_buffers | named_params
    return sorted(bufs_and_params)
