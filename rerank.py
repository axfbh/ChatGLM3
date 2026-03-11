from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, documents):
    # 构造输入格式：[CLS] query [SEP] document [SEP]
    inputs = tokenizer([f"{query} {doc}" for doc in documents], return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=-1)[:,-1].tolist() #获得相关性得分

    ranked_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return [(score, doc) for score, doc in ranked_docs]

if __name__ == "__main__":
    query = "What is the capital of France?"
    documents = ["Paris is the capital of France.", "Berlin is the capital of Germany."]
    reranked_docs = rerank(query, documents)

    print("Reranked documents:")
    for score, doc in reranked_docs:
        print(f"Score: {score:.4f}, Document: {doc}")