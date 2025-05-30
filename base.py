import os
import time
import copy
import sys
import torch
from torch import nn
import logging
from utils import *
from models import MainModel
import pdb 
from tqdm import tqdm 
import time


class BaseModel(nn.Module):
    def __init__(self, device, lr=1e-3, patience=5, result_dir='./results', hash_id=None, **kwargs):
        super(BaseModel, self).__init__()
        
        self.epochs = kwargs['epochs']
        self.lr = lr
        self.device = device
        self.model_save_dir = os.path.join(kwargs['model_save_dir'], hash_id) 
        self.patience = patience 

        if kwargs['model'] == 'all':
            self.model = MainModel(self.device, **kwargs)

        else:
            print("Please select a valid model")
            sys.exit(1)
        self.model.to(device)


    def save_model(self, state, file=None):
        if file is None: file = os.path.join(self.model_save_dir, "model.ckpt")
        try:
            torch.save(state, file, _use_new_zipfile_serialization=False)
        except:
            torch.save(state, file)
    

    def load_model(self, model_save_file=""):
        self.model.load_state_dict(torch.load(model_save_file, map_location=self.device))
    

    def evaluate(self, test_loader, datatype ="Test"):
        self.model.eval()
        batch_cnt, epoch_loss = 0, 0.0

        top_n = [0 for _ in range(5)]
 
        with torch.no_grad():
            for batched_graph, fault_gts, node_counts in test_loader: 

                # start_time = time.time() 
                res = self.model.forward(batched_graph.to(self.device), fault_gts, node_counts)
                # end_time = time.time() 
                # execution_time = end_time - start_time
                # print(f"Execution time: {execution_time:.6f} seconds")

                
                for idx, ranked_list in enumerate(res['y_pred']): 
                    for k in range(1, 6):
                        top_k = ranked_list[:k]
                        for candidate in top_k:
                            if candidate in fault_gts[idx]:
                                top_n[k-1] += 1
                                break

                epoch_loss += res["loss"].item()
                batch_cnt += 1

        epoch_loss = epoch_loss / batch_cnt

        eval_results = {
                "loss": epoch_loss,
                "top_n": top_n  
                }  


        logging.info(
            "{} -- loss: {:.4f}, {}".format(
                datatype,
                epoch_loss,
                ", ".join([f"Top-{i+1}: {v}" for i, v in enumerate(top_n)])
            )
        )
        return eval_results

    def fit(self, train_loader, test_loader=None, evaluation_epoch=1):

        best_value, coverage, best_state, eval_res = -1, None, None, None 
        pre_loss, worse_count = float("inf"), 0 

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(1, self.epochs+1):
            self.model.train()
            batch_cnt, epoch_loss = 0, 0.0
            epoch_time_start = time.time()

            for batched_graph, fault_gts, node_counts in train_loader: 
                optimizer.zero_grad()
                res = self.model.forward(batched_graph.to(self.device), fault_gts, node_counts)
                loss = res['loss']
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_cnt += 1
            epoch_time_elapsed = time.time() - epoch_time_start
            epoch_loss = epoch_loss / batch_cnt
            logging.info("Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, self.epochs, epoch_loss, epoch_time_elapsed))
    
            if epoch_loss > pre_loss:
                worse_count += 1
                if self.patience > 0 and worse_count >= self.patience:
                    logging.info("Early stop at epoch: {}".format(epoch))
                    break
            else: 
                worse_count = 0
            pre_loss = epoch_loss

            ####### Evaluate test data during training #######
            if (epoch+1) % evaluation_epoch == 0:
                test_results = self.evaluate(test_loader, datatype="Test")

                current_topn_sum = sum(test_results["top_n"])  

                if current_topn_sum  > best_value:
                    best_value, eval_res, coverage  = current_topn_sum , test_results, epoch

        logging.info("* Best result got at epoch {} with Top-n: {}".format(
            coverage, 
            ", ".join([f"Top-{i+1}: {v:.4f}" for i, v in enumerate(eval_res["top_n"])])
        ))

        return eval_res, coverage 

    

