from sentence_transformers import SentenceTransformer, CrossEncoder
import math
import numpy as np


# ---------- Encoder Model ----------
encoder_model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True,
    device="mps"
)


# ---------- Reranking Model ----------
reranking_model = CrossEncoder(
    "BAAI/bge-reranker-large",
    device="mps"
)


def Retrieve(query,top_k,index,chunk_text,source):
  prefix = ["search_query: "+query]
  query_embedding = encoder_model.encode(prefix,normalize_embeddings=True).astype(np.float32)

  scores,indices = index.search(query_embedding,top_k)

  result=[]
  for score,idx in zip(scores[0],indices[0]):
    result.append({
        "score":score,
        "text":chunk_text[idx],
        "source":source
    })
  return result


def Reranking(question, results, top_k=5, alpha=0.5):
    """
    Reranks results by blending vector scores and normalized BGE scores.

    alpha = 1.0 -> Uses ONLY vector database scores.
    alpha = 0.0 -> Uses ONLY BGE reranker scores.
    alpha = 0.5 -> Perfect 50/50 balance (Recommended safety net).
    """
    if not results:
        return []

    pairs = [(question, r['text']) for r in results]
    raw_scores = reranking_model.predict(pairs)

    for i, result in enumerate(results):
        # 1. Apply Sigmoid to map raw BGE logits to a stable 0 to 1 range
        bge_logits = float(raw_scores[i])
        normalized_bge = 1 / (1 + math.exp(-bge_logits))
        result['rerank_score'] = normalized_bge

        # 2. Get your original vector database score (adjust 'score' key if needed)
        vector_score = result.get('score', result.get('similarity', 0.5))

        # 3. Blend them! Do not throw away your perfect Top-1 vector match
        result['final_blended_score'] = (alpha * vector_score) + ((1 - alpha) * normalized_bge)

    # Put highest blended scores first
    reranked = sorted(results, key=lambda x: x['final_blended_score'], reverse=True)

    return reranked[:top_k]
