from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pdb 


model_name =  "Salesforce/codegen-350M-mono"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def get_context_embedding(codebase: str):
    inputs = tokenizer(codebase, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(device).long()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # shape: [1, seq_len, hidden_dim] #embedding dimension:1024

    # average-pooling 
    average_embedding = last_hidden.mean(dim=1).squeeze(0)  # shape: [hidden_dim]

    return average_embedding

def get_batched_context_embeddings(codebases):

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    inputs = tokenizer(
        codebases, 
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True  
    )
    input_ids = inputs["input_ids"].to(device).long()
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # shape: [batch_size, seq_len, hidden_dim]

    attention_mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    masked_hidden = last_hidden * attention_mask
    summed = masked_hidden.sum(dim=1)
    counts = attention_mask.sum(dim=1)
    average_embeddings = summed / counts  

    return average_embeddings  
