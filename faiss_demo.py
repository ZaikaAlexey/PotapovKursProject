import faiss
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from relation_model import SyntheticDataset, BiLSTMRelation


def build_and_search():
    checkpoint_path = "outputs/model.pth"
    if not os.path.exists(checkpoint_path):
        print("Ошибка: Модель не найдена. Сначала запустите обучение.")
        return

    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = checkpoint["vocab"]
    cfg = checkpoint["cfg"]

    model = BiLSTMRelation(
        vocab_size=len(vocab),
        emb_dim=cfg["emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_classes=len(checkpoint["id2label"])
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # 1. Загружаем данные для индексации (валидационная выборка)
    val_data = SyntheticDataset(n=500, seed=2)
    texts = [s[0] for s in val_data.samples]
    X_val, _ = val_data.encode()

    # 2. Извлекаем эмбеддинги (выход LSTM без финального слоя)
    # Для этого создадим "обрубленную" модель или перехватим выход LSTM
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(X_val), 32):
            batch_x = X_val[i:i + 32]
            emb = model.embedding(batch_x)
            lstm_out, _ = model.lstm(emb)
            pooled, _ = torch.max(lstm_out, dim=1)
            embeddings.append(pooled.numpy())

    embeddings = np.vstack(embeddings).astype('float32')

    # Нормализация для косинусного расстояния (в FAISS скалярное произведение нормализованных = косинус)
    faiss.normalize_L2(embeddings)

    # 3. Строим индекс HNSW
    d = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 80
    index.add(embeddings)
    print(f"FAISS индекс построен: {index.ntotal} векторов; dim={d}")

    # 4. Поиск
    query_text = "[E1] Боб [/E1] родился в [E2] Париже [/E2]"
    print(f"\nПоисковый запрос: {query_text}")

    q_tokens = query_text.lower().split()
    q_ids = [vocab.get(t, 1) for t in q_tokens]
    max_len = 15
    if len(q_ids) < max_len:
        q_ids = q_ids + [0] * (max_len - len(q_ids))
    else:
        q_ids = q_ids[:max_len]

    q_tensor = torch.tensor([q_ids], dtype=torch.long)

    with torch.no_grad():
        q_emb = model.embedding(q_tensor)
        q_lstm, _ = model.lstm(q_emb)
        q_pooled, _ = torch.max(q_lstm, dim=1)
        q_vec = q_pooled.numpy().astype('float32')

    faiss.normalize_L2(q_vec)

    # Ищем топ-3
    D, I = index.search(q_vec, k=3)

    print("\nТоп-3 похожих предложения:")
    for rank, idx in enumerate(I[0]):
        print(f"{rank + 1}. [Расстояние: {D[0][rank]:.4f}] {texts[idx]}")


if __name__ == "__main__":
    build_and_search()