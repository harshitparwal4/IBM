from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def load_model():
    model_name = "distilbert-base-uncased-distilled-squad"
    
    # Loading the Question Answering model
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load a sentence transformer model for embedding generation
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Use a model suitable for sentence embeddings
    
    return model, tokenizer, embedding_model

def create_vector_store(text, embedding_model):
    sentences = text.split(". ")  # Splitting text into sentences
    embeddings = embedding_model.encode(sentences)  # Generate sentence embeddings
    embeddings = np.array(embeddings)  # Convert to numpy array

    # Create FAISS index for the sentence embeddings
    dim = embeddings.shape[1]  # Dimensionality of the embeddings
    index = faiss.IndexFlatL2(dim)  # Using L2 (Euclidean) distance
    index.add(embeddings)  # Adding embeddings to the index
    
    return {"sentences": sentences, "index": index, "embedding_model": embedding_model}
    

def get_answer(model, tokenizer, vector_store, question):
    sentences = vector_store["sentences"]
    index = vector_store["index"]
    embedding_model = vector_store["embedding_model"]

    # Get the embedding for the question
    q_embedding = embedding_model.encode([question])
    q_embedding = np.array(q_embedding)  # Convert to numpy array for FAISS search

    # Perform FAISS search for the most relevant sentence
    D, I = index.search(q_embedding, k=1)
    context = sentences[I[0][0]]  # Get the most relevant sentence

    # Tokenize the input for the model
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True)

    # Get the answer from the QA model
    outputs = model(**inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    start = int(start_scores.argmax())  # Start index of the answer span
    end = int(end_scores.argmax()) + 1  # End index of the answer span

    # Convert token indices to string
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end]))
    return answer