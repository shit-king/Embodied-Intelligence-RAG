from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_BATCH_SIZE, EMBEDDING_DEVICE, EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
    model.max_seq_length = 512
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_model().encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 100,
    ).tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]
