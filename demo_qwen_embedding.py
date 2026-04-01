import torch
import torch.nn.functional as F

from torch import Tensor
from modelscope import AutoModelForCausalLM, AutoTokenizer

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:,-1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        seqence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size,device=last_hidden_states.device), seqence_lengths]

def get_detail_instruct(task_description, query):
    return f"Instruction: {task_description}\nQuery: {query}"


task = "Given a web search query, retrieve relevant passages that answer the query."

query = [
    get_detail_instruct(task, "What is the capital of China?"),
    get_detail_instruct(task, "Explain gravity"),
]

documents = [
    "The capital of China is Beijing.",
    "Gravity is the force of attraction between two objects with mass.",
]

input_texts = query + documents

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Embedding-0.6B")

max_length = 8192

batch_dict = tokenizer(
    input_texts, 
    padding=True, 
    truncation=True, 
    max_length=max_length, 
    return_tensors="pt"
).to(model.device)

outputs = model.model(**batch_dict)
embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:2] @ embeddings[2:].T)
print(scores.tolist())