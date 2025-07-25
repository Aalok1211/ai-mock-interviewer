import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

RUBRIC_WEIGHTS = {
    "formula_accuracy": 0.4,
    "conceptual_depth": 0.2,
    "efficiency": 0.15,
    "communication": 0.15,
    "problem_solving": 0.1,
}


def load_rubric(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def score_answer(candidate_answer: str, ideal_answer: str, rubric_df: pd.DataFrame) -> dict:
    """Returns a dict with rubric categories and 1-5 scores."""
    # Simple TF-IDF cosine similarity for formula & concept overlap
    tfidf = TfidfVectorizer().fit([candidate_answer, ideal_answer])
    vecs = tfidf.transform([candidate_answer, ideal_answer])
    similarity = cosine_similarity(vecs[0], vecs[1])[0][0]

    # Original:
    # similarity_score = int(np.clip(round(similarity * 5), 1, 5))
    # Revised:
    norm_similarity = (similarity + 1) / 2  # normalize cosine similarity to 0–1
    scaled_score = round(norm_similarity * 5)
    similarity_score = max(0, min(5, scaled_score))

    scores = {
        "formula_accuracy": similarity_score,
        "conceptual_depth": similarity_score,
        "efficiency": max(0, similarity_score - 1),  # allow 0 as min
        "communication": similarity_score,
        "problem_solving": max(0, similarity_score - 2),
    }
    weighted = sum(scores[k] * w for k, w in RUBRIC_WEIGHTS.items()) / sum(RUBRIC_WEIGHTS.values())
    scores["total"] = round(weighted, 2)
    return scores