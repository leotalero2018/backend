import json
from typing import List
from langchain_community.embeddings import HuggingFaceHubEmbeddings

class HuggingFaceHubEmbeddingsBatched(HuggingFaceHubEmbeddings):
    batch_size: int = 32

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        texts = [text.replace("\n", " ") for text in texts]
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            responses = self.client.post(
                json={"inputs": batch, "truncate": True}, task=self.task
            )
            embeddings.extend(json.loads(responses.decode()))

        return embeddings
