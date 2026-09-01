from __future__ import annotations

from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_texts = [doc.text for doc in documents]
        self.matrix = self.vectorizer.fit_transform(self.doc_texts)

    def retrieve(self, question: str, top_k: int = 3):
        if not self.documents:
            return []
        query_vector = self.vectorizer.transform([question])
        sims = cosine_similarity(query_vector, self.matrix).flatten()
        ranked_indices = sims.argsort()[::-1]
        results = []
        for idx in ranked_indices[:top_k]:
            if sims[idx] > 0:
                results.append(self.documents[int(idx)])
        return results
