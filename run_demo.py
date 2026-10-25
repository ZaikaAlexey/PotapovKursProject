import argparse
from relation_model import TrainConfig, train_and_eval

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Запуск обучения модели извлечения отношений")
    ap.add_argument("--epochs", type=int, default=15, help="Количество эпох обучения")
    ap.add_argument("--batch-size", type=int, default=32, help="Размер батча")
    ap.add_argument("--emb-dim", type=int, default=50, help="Размерность эмбеддингов")
    ap.add_argument("--hidden-dim", type=int, default=64, help="Скрытая размерность LSTM")
    ap.add_argument("--lr", type=float, default=1e-3, help="Скорость обучения")
    ap.add_argument("--device", type=str, default="cpu", help="Устройство (cpu или cuda)")
    ap.add_argument("--out-dir", type=str, default="outputs", help="Директория для сохранения")
    args = ap.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        device=args.device,
        out_dir=args.out_dir
    )

    print("Начало обучения...")
    metrics = train_and_eval(cfg)
    print("Обучение завершено. Итоговые метрики:", metrics)