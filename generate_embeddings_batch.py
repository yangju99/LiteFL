import os
import json
import re
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm 
from utils import *
from llm import get_context_embedding, get_batched_context_embeddings

def read_json(filepath):
    if os.path.exists(filepath):
        assert filepath.endswith('.json')
        with open(filepath, 'r') as f:
            return json.loads(f.read())
    else: 
        return None


def generate_chunks(data_dir, save_dir, embedding_size, batch_size=3):

    chunks = [] 
    
    for entry in tqdm(os.listdir(data_dir), desc="Generating embeddings for buggy versions"):
        bug_id = entry  # example: Chart_1
        print(f"[•] Processing: {bug_id}")

        snippet_path = os.path.join(data_dir, entry, "snippet.json")
        covered_methods = read_json(snippet_path)
        method_num = len(covered_methods)

        id_to_name, name_to_id, id_to_code, id_to_comment = {}, {}, {}, {}
        fault_ids = []

        next_id = 0
        for covered_method in covered_methods:

            signature = covered_method["signature"]
            method_name = signature.split("(")[0]

            method_id = next_id
            next_id += 1

            name_to_id[method_name] = method_id
            id_to_name[method_id] = method_name

            id_to_code[method_id] = covered_method["snippet"]
            id_to_comment[method_id] = covered_method.get("comment", "").strip()

            if covered_method.get("is_bug", False):
                fault_ids.append(method_id)
        
        chunk = {} 
        chunk['project_id'] = entry
        chunk['code_vector'] = np.zeros((method_num, embedding_size))
        chunk['comment_vector'] = np.zeros((method_num, embedding_size))


        sorted_ids = sorted(id_to_code.keys())
        code_strs = [id_to_code[idx] for idx in sorted_ids]
        comment_strs = [id_to_comment[idx] for idx in sorted_ids]

        # ---------- code_vector ----------
        for i in range(0, len(code_strs), batch_size):
            batch_code = code_strs[i:i+batch_size]
            batch_ids = sorted_ids[i:i+batch_size]

            batch_embeddings = get_batched_context_embeddings(batch_code).detach().cpu().numpy()

            for idx, method_id in enumerate(batch_ids):
                chunk['code_vector'][method_id] = batch_embeddings[idx]

        # ---------- comment_vector ----------
        for i in range(0, len(comment_strs), batch_size):
            batch_comment = comment_strs[i:i+batch_size]
            batch_ids = sorted_ids[i:i+batch_size]

            non_empty_inputs = []
            non_empty_indices = []
            for idx, text in enumerate(batch_comment):
                if text != "":
                    non_empty_inputs.append(text)
                    non_empty_indices.append(idx)

            if non_empty_inputs:
                batch_embeddings = get_batched_context_embeddings(non_empty_inputs).detach().cpu().numpy()
            else:
                batch_embeddings = []

            for local_idx, method_id in enumerate(batch_ids):
                if local_idx in non_empty_indices:
                    emb_idx = non_empty_indices.index(local_idx)
                    chunk['comment_vector'][method_id] = batch_embeddings[emb_idx]
                else:
                    chunk['comment_vector'][method_id] = np.zeros((embedding_size,))


        chunk['fault_label'] = fault_ids
        chunks.append(chunk)

    os.makedirs(save_dir, exist_ok=True)

    pkl_path = os.path.join(save_dir, "all.pkl")
    with open(pkl_path, 'wb') as pkl:
        pickle.dump(chunks, pkl)

    return chunks



if __name__ == "__main__":

    data_dir = "./defects4j"
    save_dir = "./chunks"

    sample_embedding = get_context_embedding("def add(a,b): return a+b")
    embedding_size = sample_embedding.shape[0]

    batch_size = 10


    generate_chunks(data_dir, save_dir, embedding_size, batch_size)





