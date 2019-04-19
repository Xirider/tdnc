# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
from pathlib import Path
import random
from io import open
import pickle
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import random

import tarfile
import requests




logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
def maybe_download(directory, filename, uri):
  
    filepath = os.path.join(directory, filename)
    if not os.path.exists(directory):
        logger.info(f"Creating new dir: {directory}")
        os.makedirs(directory)
    if not os.path.exists(filepath):
        logger.info("Downloading und unpacking file, as file does not exist yet")
        r = requests.get(uri, allow_redirects=True)
        open(filepath, "wb").write(r.content)

    return filepath
    




def load_weights_from_state(model, state_dict):

    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')
    start_prefix = ''
    if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
        start_prefix = 'bert.'
    load(model, prefix=start_prefix)
    if len(missing_keys) > 0:
        logger.info("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        logger.info("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                            model.__class__.__name__, "\n\t".join(error_msgs)))
    return model
