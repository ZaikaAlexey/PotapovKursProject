import streamlit as st
import torch
import os
from relation_model import BiLSTMRelation

st.set_page_config(page_title="Извлечение отношений", layout="centered")
st.title("Извлечение отношений из текста (BiLSTM)")
st.markdown("Введите предложение с маркерами сущностей `[E1]` и `[E2]`, чтобы модель определила связь между ними.")


@st.cache_resource
def load_model():
    checkpoint_path = "outputs/model.pth"
    if not os.path.exists(checkpoint_path):
        return None, None, None

    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    vocab = checkpoint["vocab"]
    id2label = checkpoint["id2label"]
    cfg = checkpoint["cfg"]

    model = BiLSTMRelation(
        vocab_size=len(vocab),
        emb_dim=cfg["emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_classes=len(id2label)
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, vocab, id2label


model, vocab, id2label = load_model()

if model is None:
    st.error("Модель не найдена! Сначала запустите `run_demo.py` для обучения.")
else:
    # Поле ввода
    default_text = "[E1] Алиса [/E1] работает в [E2] Google [/E2]"
    text_input = st.text_area("Введите текст:", value=default_text, height=100)

    if st.button("Извлечь отношение"):
        # Токенизация (как при обучении)
        tokens = text_input.lower().split()
        ids = [vocab.get(t, 1) for t in tokens]  # 1 = <unk>

        max_len = 15
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))
        else:
            ids = ids[:max_len]

        x = torch.tensor([ids], dtype=torch.long)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred_id = torch.argmax(probs).item()
            confidence = probs[pred_id].item()

        pred_label = id2label[pred_id]

        # Красивый вывод
        st.markdown("### Результат анализа:")
        if pred_label == "no_relation":
            st.warning(f"**Отношение не найдено** (Уверенность: {confidence:.2%})")
        else:
            st.success(f"**Найденное отношение:** `{pred_label}` (Уверенность: {confidence:.2%})")

        # Детальные вероятности
        st.markdown("#### Распределение вероятностей:")
        for i, label in id2label.items():
            st.progress(float(probs[i]), text=f"{label}: {probs[i]:.2%}")