from audiocraft.models import MusicGen
from audiocraft.modules.conditioners import ConditioningAttributes
from peft import LoraConfig, get_peft_model
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
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
LR = 5e-5
EPOCHS = 3
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
    # Приводим LoRA-адаптеры к fp32: GradScaler требует fp32-параметры,
    # autocast сам кастит их в fp16 для forward-pass
    for param in model.lm.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    trainable = sum(p.numel() for p in model.lm.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.lm.parameters())
    print(f"Обучаемых параметров: {trainable:,} из {total:,} ({100 * trainable / total:.2f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.lm.parameters() if p.requires_grad], lr=LR
    )
    scaler = GradScaler()

    dataset = AudioDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    for epoch in range(EPOCHS):
        model.lm.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            wav = batch['wav'].to(device)
            if wav.shape[1] > 1:
                wav = wav.mean(dim=1, keepdim=True)  # стерео → моно
            with torch.autocast(device_type=device):
                codes = model.compression_model.encode(wav)[0]
                conditions = [ConditioningAttributes(text={'description': c}) for c in batch['caption']]
                out = model.lm.compute_predictions(codes=codes, conditions=conditions)

            # Получаем delay-patterned последовательность как цели
            T = codes.shape[-1]
            pattern = model.lm.pattern_provider.get_pattern(T)
            sequence_codes, _, _ = pattern.build_pattern_sequence(
                codes, model.lm.special_token_id
            )

            logits = out.logits.float()  # [B, K, S_logits, card]
            B, K, S_logits, card = logits.shape
            S_seq = sequence_codes.shape[2]
            S = min(S_logits, S_seq - 1)
            pred = logits[:, :, :S].reshape(-1, card)
            target = sequence_codes[:, :, 1:S + 1].reshape(-1).long()

            # Хвостовые позиции delay-паттерна дают NaN логиты — маскируем их
            nan_positions = pred.isnan().any(dim=-1)
            target = target.masked_fill(nan_positions, model.lm.special_token_id)
            pred = pred.nan_to_num(0.0)

            loss = F.cross_entropy(
                pred, target, ignore_index=model.lm.special_token_id
            ) / GRAD_ACCUM

            scaler.scale(loss).backward()
            total_loss += loss.item()

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.lm.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
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
