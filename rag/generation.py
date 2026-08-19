from models.retrieving_models import Reranking,Retrieve
from models.llm import Generate_LLM_Response
from models.tts import Generate_Speech
import json
import faiss

# Load FAISS index once
index = faiss.read_index("rag_data/index.faiss")

# Load chunks once
with open("rag_data/chunks.json", "r", encoding="utf-8") as f:
    chunk_text = json.load(f)

def Generate_Reply(query,top_k):
    retrieval = Retrieve(query=query,top_k=top_k,chunk_text=chunk_text,index=index,source="terms and conditions")
    reranking = Reranking(question=query,results=retrieval)
    print(reranking[0]['text'])
    final_response = Generate_LLM_Response(query=query,context=reranking[0]['text'])

    if not final_response:
        return None
    
    return final_response