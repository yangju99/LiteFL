from torch.utils.data import Dataset, DataLoader
import torch
import dgl
from utils import * 
import pickle
import sys
import logging
from base import BaseModel
import time
from utils import *
import pandas as pd 
import pdb 
from tqdm import tqdm 
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))


class chunkDataset(Dataset):
    def __init__(self, chunks):
        self.data = []
        self.idx2id = {}
        for idx, chunk  in enumerate(chunks):
            self.idx2id[idx] = chunk['project_id']

            graph = dgl.graph(([], []),num_nodes = len(chunk['code_vector']))

            graph.ndata["code_vector"] = torch.FloatTensor(chunk["code_vector"])
            graph.ndata["comment_vector"] = torch.FloatTensor(chunk["comment_vector"])
 
            fault_gts = chunk['fault_label']

            self.data.append((graph, fault_gts))
                
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def leave_one_out_project(params):

    with open(params["data_path"], 'rb') as te:
        chunks = pickle.load(te)
    te.close()

    params["code_dim"] = chunks[0]['code_vector'].shape[1]
    params["comment_dim"] = chunks[0]['comment_vector'].shape[1]


    device = get_device(params["check_device"])

    hash_id = dump_params(params)

    params["hash_id"] = hash_id
    print("hash_id: ", hash_id)
    
    total_top_n = [0 for _ in range(5)]

    for i in tqdm(range(len(chunks)), desc="Doing Leave-one-out cross validation") :
        test_data = chunkDataset([chunks[i]])
        train_data = chunkDataset(chunks[:i] + chunks[i+1:])

        train_dl = DataLoader(train_data, batch_size = params['batch_size'], shuffle=True, collate_fn=collate, pin_memory=True)
        test_dl = DataLoader(test_data, batch_size = params['batch_size'], shuffle=False, collate_fn=collate, pin_memory=True)

        model = BaseModel(device, lr = params["learning_rate"], **params)

         # Train model
        eval_res, converge = model.fit(train_dl, test_dl, evaluation_epoch= params['evaluation_epoch'])
        
        for idx, value in enumerate(eval_res['top_n']):
            total_top_n[idx] += value 
        

    dump_scores(params["model_save_dir"], hash_id, total_top_n)
        
    logging.info("Current hash_id {}".format(hash_id))


# Instantiate your Dataset and DataLoader
############################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the fault localization model")
    parser.add_argument('--data_path', type=str, default="/root/project/processed_data/Closure.pkl", help='Path to the dataset file')
    parser.add_argument('--random_seed', type=int, default=12345, help='Random seed for reproducibility')

    args = parser.parse_args()

    random_seed = args.random_seed
    batch_size = 1
    epochs = 30
    evaluation_epoch = 1
    learning_rate = 0.001 
    model = "all"
    result_dir = "./results"
    code_dim = None
    comment_dim = None

    seed_everything(random_seed)

    params = {
            'batch_size': batch_size,
            'epochs' : epochs,
            'evaluation_epoch': evaluation_epoch,
            'learning_rate': learning_rate, 
            'model': 'all',
            'model_save_dir': result_dir,
            'code_dim': code_dim,
            'comment_dim': comment_dim,
            'check_device': "gpu",
            'classification_hiddens': [128, 64],
            'data_path' : args.data_path
    }     

    leave_one_out_project(params)






