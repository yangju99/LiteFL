import torch
from torch import nn
from dgl.nn.pytorch import GATv2Conv
import pdb 
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        alpha: scalar (e.g., 5.0 for class 1 emphasis) or Tensor([alpha_0, alpha_1])
        gamma: focusing parameter
        reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  

        if isinstance(self.alpha, torch.Tensor):
            at = self.alpha.to(logits.device)[targets]
        else:
            at = self.alpha

        focal_loss = at * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



class CommentModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CommentModel, self).__init__()
        self.embedder = nn.Linear(in_dim, out_dim) 
    def forward(self, paras: torch.tensor): 
        """
        Input:
            paras: mu with length of event_num
        """
        return self.embedder(paras)


class CodeModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CodeModel, self).__init__()
        self.embedder = nn.Linear(in_dim, out_dim) 
    def forward(self, paras: torch.tensor): 
        """
        Input:
            paras: mu with length of event_num
        """
        return self.embedder(paras)


class MultiSourceEncoder(nn.Module):
    def __init__(self, device, code_out_dim=512, comment_out_dim=512, fuse_dim=512, **kwargs):
        super(MultiSourceEncoder, self).__init__()

        self.code_model = CodeModel(kwargs['code_dim'], code_out_dim)  
        self.comment_model = CommentModel(kwargs['comment_dim'], comment_out_dim)  
        fuse_in = code_out_dim + comment_out_dim

        if not fuse_dim % 2 == 0: fuse_dim += 1

        self.fuse = nn.Linear(fuse_in, fuse_dim)
       
        self.activate = nn.GLU()
        self.feat_out_dim = int(fuse_dim // 2)
    
    def forward(self, graph):
        code_embedding = self.code_model(graph.ndata["code_vector"]) 
        comment_embedding = self.comment_model(graph.ndata["comment_vector"]) 

        feature = self.activate(self.fuse(torch.cat((code_embedding, comment_embedding), dim=-1)))

        return feature

class FullyConnected(nn.Module):
    def __init__(self, in_dim, out_dim, linear_sizes):
        super(FullyConnected, self).__init__()
        layers = []
        for i, hidden in enumerate(linear_sizes):
            input_size = in_dim if i == 0 else linear_sizes[i-1]
            layers += [nn.Linear(input_size, hidden), nn.ReLU()]
        layers += [nn.Linear(linear_sizes[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)


import numpy as np
class MainModel(nn.Module):
    def __init__(self, device, debug=False, **kwargs):
        super(MainModel, self).__init__()

        self.device = device

        self.encoder = MultiSourceEncoder(device, debug=debug, **kwargs)

        self.classifier = FullyConnected(self.encoder.feat_out_dim, 2, kwargs['classification_hiddens']).to(device)

        self.criterion = FocalLoss(gamma=2.0) 
        self.get_prob = nn.Softmax(dim=-1)

    def forward(self, graph, fault_indexs, node_counts):
        """
        Pointwise loss version.
        """
        embeddings = self.encoder(graph)  

        logits = self.classifier(embeddings)

        y_true = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)

        offset = 0
        for faults, count in zip(fault_indexs, node_counts):
            for idx in faults:
                y_true[offset + idx] = 1
            offset += count

        loss = self.criterion(logits, y_true)

        node_probs = self.get_prob(logits.detach()).cpu().numpy()

        y_pred = self.inference(node_probs, node_counts)

        return {
            'loss': loss,
            'y_pred': y_pred
        }


    def inference(self, node_probs, node_counts):
        """
        node_probs: [total_nodes, 2]
        node_counts: list of node counts per graph in batch
        Returns:
            list of ranked indices per graph
        """
        fault_probs = node_probs[:, 1]  
        results = []
        start = 0
        for count in node_counts:
            sub_probs = fault_probs[start:start+count]
            ranked = sub_probs.argsort()[::-1].tolist()
            results.append(ranked)  
            start += count
        return results


