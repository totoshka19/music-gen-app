from audiocraft.models import MusicGen
from peft import LoraConfig, get_peft_model
import torch
import json
from torch.utils.data import Dataset, DataLoader
import torchaudio
from pathlib import Path

# ---- НАСТРОЙКИ ----
MODEL_NAME = 'facebook/musicgen-medium'
DATASET_DIR = 'dataset/processed'
METADATA_FILE = 'dataset/metadata.jsonl'
OUTPUT_DIR = 'checkpoints'
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 1e-4
EPOCHS = 10
# -------------------


class AudioDataset(Dataset):
    def __init__(self):
        with open(METADATA_FILE, encoding='utf-8') as f:
            self.items = [json.loads(line) for line in f if line.strip()]
        print(f"Датасет: {len(self.items)} сегментов")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        wav, sr = torchaudio.load(f"{DATASET_DIR}/{item['audio']}")
        if sr != 32000:
            wav = torchaudio.functional.resample(wav, sr, 32000)
        return {'wav': wav, 'caption': item['caption']}


def main():
    print("Загружаем модель...")
    model = MusicGen.get_pretrained(MODEL_NAME)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Устройство: {device}")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model.lm = get_peft_model(model.lm, lora_config)

    trainable = sum(p.numel() for p in model.lm.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.lm.parameters())
    print(f"Обучаемых параметров: {trainable:,} из {total:,} ({100 * trainable / total:.2f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.lm.parameters() if p.requires_grad], lr=LR
    )

    dataset = AudioDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    for epoch in range(EPOCHS):
        model.lm.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            with torch.autocast(device):
                codes = model.compression_model.encode(batch['wav'].to(device))[0]
                conditions = [{'description': [c]} for c in batch['caption']]
                out = model.lm.compute_predictions(codes=codes, conditions=conditions)
                loss = out.loss / GRAD_ACCUM

            loss.backward()
            total_loss += loss.item()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.lm.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % 50 == 0:
                avg = total_loss / (step + 1)
                print(f"Эпоха {epoch + 1}/{EPOCHS}, шаг {step}, loss={avg:.4f}")

        save_path = f"{OUTPUT_DIR}/epoch_{epoch + 1}"
        model.lm.save_pretrained(save_path)
        print(f"Чекпоинт сохранён: {save_path}")

    print("Обучение завершено!")


if __name__ == '__main__':
    main()
