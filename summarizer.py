from sentence_transformers import SentenceTransformer, util

def summarize_text(vector_store, max_sentences=3):
    sentences = vector_store["sentences"]
    embedding_model = vector_store["embedding_model"]
    embeddings = embedding_model.encode(sentences)

    avg_embedding = sum(embeddings) / len(embeddings)
    scores = [util.cos_sim([avg_embedding], [emb])[0][0].item() for emb in embeddings]
    ranked_sentences = sorted(zip(scores, sentences), reverse=True)
    
    summary = " ".join([sent for _, sent in ranked_sentences[:max_sentences]])
    return summary