import os
import logging
import pickle
import torch
import numpy as np
import random
import json
import logging
import dgl
import inspect
import pdb 

def read_json(filepath):
    if os.path.exists(filepath):
        assert filepath.endswith('.json')
        with open(filepath, 'r') as f:
            return json.loads(f.read())
    else: 
        return None

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(gpu):
    if gpu and torch.cuda.is_available():
        logging.info("Using GPU...")
        return torch.device("cuda")
    logging.info("Using CPU...")
    return torch.device("cpu")


def collate(data):
    graphs, fault_gts = map(list, zip(*data))
    batched_graph = dgl.batch(graphs)
    node_counts = [g.num_nodes() for g in graphs]

    return batched_graph, fault_gts, node_counts  


def save_logits_as_dict(logits, keys, filename):
    """
    Saves a list of tensors as a dictionary with variable names as keys and tensor values as dictionary values.
    """
    frame = inspect.currentframe().f_back
    tensor_dict = {}
    
    for logit in logits:
        names = [name for name, var in frame.f_locals.items() if torch.is_tensor(var) and var is tensor and not name.startswith('_')]
        
        if names:
            tensor_dict[names[0]] = logit
            
    return tensor_dict

import hashlib
def dump_params(params):
    hash_id = hashlib.md5(str(sorted([(k, v) for k, v in params.items()])).encode("utf-8")).hexdigest()[0:8]

    basename = os.path.basename(params['data_path']) 
    name = os.path.splitext(basename)[0]
    hash_id = name + "_" + hash_id

    result_dir = os.path.join(params["model_save_dir"], hash_id)
    os.makedirs(result_dir, exist_ok=True)

    json_pretty_dump(params, os.path.join(result_dir, "params.json"))

    log_file = os.path.join(result_dir, "running.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return hash_id

from datetime import datetime, timedelta
def dump_scores(result_dir, hash_id, total_top_n):
    with open(os.path.join(result_dir, 'experiments.txt'), 'a+') as fw:
        fw.write(hash_id+': '+(datetime.now()+timedelta(hours=8)).strftime("%Y/%m/%d-%H:%M:%S")+'\n')
        fw.write("* Test result -- " + str(total_top_n))
        fw.write('{}{}'.format('='*40, '\n'))


def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(obj,fw, sort_keys=True, indent=4, separators=(",", ": "), ensure_ascii=False)


