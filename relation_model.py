import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Генератор синтетических данных
class SyntheticDataset(Dataset):
    def __init__(self, n=1000, seed=42):
        self.n = n
        random.seed(seed)
        self.samples = []

        # Шаблоны предложений с маркерами сущностей
        templates = {
            "works_at": [
                "[E1] Алиса [/E1] работает в [E2] Google [/E2]",
                "[E1] Боб [/E1] является сотрудником [E2] Яндекс [/E2]",
                "[E1] Мария [/E1] трудится в [E2] Сбер [/E2]"
            ],
            "born_in": [
                "[E1] Боб [/E1] родился в [E2] Париже [/E2]",
                "[E1] Алиса [/E1] появилась на свет в [E2] Москве [/E2]",
                "[E1] Иван [/E1] был рожден в [E2] Казани [/E2]"
            ],
            "located_in": [
                "[E1] Эйфелева башня [/E1] находится в [E2] Париже [/E2]",
                "[E1] Кремль [/E1] расположен в [E2] Москве [/E2]",
                "[E1] Колизей [/E1] стоит в [E2] Риме [/E2]"
            ],
            "no_relation": [
                "[E1] Алиса [/E1] любит [E2] кошек [/E2]",
                "[E1] Боб [/E1] читает [E2] книгу [/E2]",
                "[E1] Мария [/E1] готовит [E2] суп [/E2]"
            ]
        }

        self.label2id = {k: i for i, k in enumerate(templates.keys())}
        self.id2label = {v: k for k, v in self.label2id.items()}

        for _ in range(n):
            rel_type = random.choice(list(templates.keys()))
            text = random.choice(templates[rel_type])
            self.samples.append((text, self.label2id[rel_type]))

        self.vocab = self._build_vocab()

    def _build_vocab(self):
        vocab = {"<pad>": 0, "<unk>": 1}
        for text, _ in self.samples:
            # Простая токенизация по пробелам (маркеры остаются как отдельные токены)
            tokens = text.lower().split()
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)
        return vocab

    def encode(self, max_len=15):
        X, y = [], []
        for text, label in self.samples:
            tokens = text.lower().split()
            ids = [self.vocab.get(t, 1) for t in tokens]
            if len(ids) < max_len:
                ids = ids + [0] * (max_len - len(ids))
            else:
                ids = ids[:max_len]
            X.append(ids)
            y.append(label)
        return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.samples[idx]


# 2. Архитектура BiLSTM
class BiLSTMRelation(nn.Module):
    def __init__(self, vocab_size, emb_dim=50, hidden_dim=64, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)  # [batch, seq_len, emb_dim]
        lstm_out, _ = self.lstm(emb)  # [batch, seq_len, hidden_dim*2]
        # Max pooling по временной размерности
        pooled, _ = torch.max(lstm_out, dim=1)  # [batch, hidden_dim*2]
        logits = self.fc(pooled)
        return logits


# 3. Конфигурация и обучение
class TrainConfig:
    def __init__(self, epochs=15, batch_size=32, emb_dim=50, hidden_dim=64, lr=1e-3, device="cpu", out_dir="outputs"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.out_dir = out_dir


def train_and_eval(cfg: TrainConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")

    # Данные
    train_data = SyntheticDataset(n=2000, seed=1)
    val_data = SyntheticDataset(n=500, seed=2)
    val_data.vocab = train_data.vocab

    X_train, y_train = train_data.encode()
    X_val, y_val = val_data.encode()
    vocab = train_data.vocab

    # Модель и оптимизатор
    model = BiLSTMRelation(len(vocab), cfg.emb_dim, cfg.hidden_dim, len(train_data.label2id)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    # Цикл обучения
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{cfg.epochs} | Loss: {avg_loss:.4f}")

    # Оценка
    model.eval()
    with torch.no_grad():
        X_val_t = X_val.to(device)
        logits = model(X_val_t)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        y_true = y_val.cpu().numpy()

    # Метрики
    macro_f1 = f1_score(y_true, preds, average='macro')
    acc = accuracy_score(y_true, preds)

    metrics = {
        "macro_f1": float(macro_f1),
        "accuracy": float(acc)
    }

    with open(os.path.join(cfg.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Матрица ошибок
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(val_data.label2id.keys()),
                yticklabels=list(val_data.label2id.keys()))
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.title('Матрица ошибок')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "confusion_matrix.png"))
    plt.close()

    # Сохранение весов
    torch.save({
        "model_state": model.state_dict(),
        "vocab": vocab,
        "label2id": train_data.label2id,
        "id2label": train_data.id2label,
        "cfg": vars(cfg)
    }, os.path.join(cfg.out_dir, "model.pth"))

    print(f"Метрики: Macro-F1 = {macro_f1:.4f}, Accuracy = {acc:.4f}")
    return metrics