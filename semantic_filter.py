# === semantic_filter.py ===

import os
from openai import OpenAI
client = OpenAI()
import numpy as np
from dotenv import load_dotenv
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from gpt_utils import num_tokens



# Load OpenAI API key
load_dotenv()
client.api_key = os.getenv("OPENAI_API_KEY")


EMBED_LIMIT = 8100  # embedding model max safe token count

# === 1. Embed any text ===
def embed_text(text: str) -> List[float]:
    """Embed text using OpenAI text-embedding-ada-002 with token-safe fallback."""
    try:
        if num_tokens(text) > EMBED_LIMIT:
            print(f"✂️ Truncating text for embedding ({num_tokens(text)} tokens → under {EMBED_LIMIT})")
            words = text.split()
            truncated = []
            for word in words:
                truncated.append(word)
                if num_tokens(" ".join(truncated)) >= EMBED_LIMIT:
                    break
            text = " ".join(truncated)

        response = client.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        return embedding

    except Exception as e:
        print(f"[Embedding Error] {e}")
        return None


# === 2. Cosine Similarity ===
def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    vec1 = np.array(vec1).reshape(1, -1)
    vec2 = np.array(vec2).reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

# === 3. Semantic Filter ===
def filter_keywords_by_semantic_similarity(
    resume_text: str,
    keywords: List[str],
    threshold: float = 0.65
) -> List[str]:
    """
    Filters job keywords by semantic similarity against the resume text.
    Keeps only keywords that semantically match the resume above the threshold.
    """
    resume_embedding = embed_text(resume_text)

    if resume_embedding is None:
        print("🛑 Embedding failed for resume. Skipping semantic filtering.")
        return keywords  # fallback if API call failed

    kept_keywords = []
    for keyword in keywords:
        keyword_embedding = embed_text(keyword)
        if keyword_embedding is None:
            continue  # skip if embedding failed
        similarity = compute_cosine_similarity(resume_embedding, keyword_embedding)
        print(f"🔎 {keyword} → Similarity: {similarity:.3f}")
        if similarity >= threshold:
            kept_keywords.append(keyword)

    return kept_keywords
