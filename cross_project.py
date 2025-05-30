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


def cross_project(params):
    data_dir = params["data_path"]
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl') and not f.startswith("all")]
    all_files.sort() 

    print("Cross-project files:", all_files)

    for i in range(len(all_files)):
        test_file = all_files[i]
        train_files = all_files[:i] + all_files[i+1:]


        with open(os.path.join(data_dir, test_file), 'rb') as f:
            test_chunks = pickle.load(f)

 
        train_chunks = []
        for fname in train_files:
            with open(os.path.join(data_dir, fname), 'rb') as f:
                train_chunks.extend(pickle.load(f))

        params["code_dim"] = train_chunks[0]['code_vector'].shape[1]
        params["comment_dim"] = train_chunks[0]['comment_vector'].shape[1]

        print(f"\n[Test File: {test_file}]")
        print(f"Train on: {[f for f in train_files]}")
        print(f"Train size: {len(train_chunks)}, Test size: {len(test_chunks)}")

        device = get_device(params["check_device"])
        hash_id = dump_params(params)
        params["hash_id"] = hash_id
        print("hash_id:", hash_id)

        
        train_dl = DataLoader(chunkDataset(train_chunks), batch_size=params['batch_size'],
                              shuffle=True, collate_fn=collate, pin_memory=True)
        test_dl = DataLoader(chunkDataset(test_chunks), batch_size=params['batch_size'],
                             shuffle=False, collate_fn=collate, pin_memory=True)

        model = BaseModel(device, lr=params["learning_rate"], **params)
        eval_res, converge = model.fit(train_dl, test_dl, evaluation_epoch=params['evaluation_epoch'])

        top_n_result = {test_file: eval_res['top_n']}

        dump_scores(params["model_save_dir"], hash_id, top_n_result)

        logging.info(f"[{test_file}] Top-N scores: {eval_res['top_n']}")


# Instantiate your Dataset and DataLoader
############################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the fault localization model")
    parser.add_argument('--data_path', type=str, default="/workspace/ase/data/codegen_350m", help='Path to the dataset file')
    parser.add_argument('--random_seed', type=int, default=12345, help='Random seed for reproducibility')

    args = parser.parse_args()

    random_seed = args.random_seed
    batch_size = 1
    epochs = 30
    evaluation_epoch = 1
    learning_rate = 0.001 
    model = "all"
    result_dir = "./cross_results"
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

    cross_project(params)






