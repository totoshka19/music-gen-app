"""Диагностика — находим где именно NaN в логитах."""
from audiocraft.models import MusicGen
from audiocraft.modules.conditioners import ConditioningAttributes
import torch
import torchaudio
import json

MODEL_NAME = 'facebook/musicgen-medium'
DATASET_DIR = 'dataset/processed'
METADATA_FILE = 'dataset/metadata.jsonl'

model = MusicGen.get_pretrained(MODEL_NAME)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(METADATA_FILE, encoding='utf-8') as f:
    item = json.loads(f.readline())
wav, sr = torchaudio.load(f"{DATASET_DIR}/{item['audio']}")
if sr != 32000:
    wav = torchaudio.functional.resample(wav, sr, 32000)
if wav.shape[0] > 1:
    wav = wav.mean(dim=0, keepdim=True)
wav = wav.unsqueeze(0).to(device)

with torch.no_grad():
    codes = model.compression_model.encode(wav)[0][:, :, :50]  # только 50 токенов

conditions = [ConditioningAttributes(text={'description': item['caption']})]

model.lm.eval()
with torch.no_grad():
    with torch.autocast(device_type=device):
        out = model.lm.compute_predictions(codes=codes, conditions=conditions)

logits = out.logits.float()  # [1, 4, S, 2048]
nan_mask = logits.isnan().any(dim=-1)  # [1, 4, S]

print(f"Logits shape: {logits.shape}")
print(f"\nNaN по кодбукам:")
for k in range(4):
    n = nan_mask[0, k].sum().item()
    first = nan_mask[0, k].nonzero(as_tuple=True)[0][:5].tolist() if n > 0 else []
    print(f"  codebook {k}: {n}/{logits.shape[2]} NaN, первые позиции: {first}")

print(f"\nNaN в любом кодбуке: {nan_mask[0].any(dim=0).sum().item()}/{logits.shape[2]} позиций")
print(f"Только в конкретных кодбуках: {nan_mask[0].sum(dim=1).tolist()}")

# Проверяем входной sequence_codes
T = codes.shape[-1]
pattern = model.lm.pattern_provider.get_pattern(T)
special_id = model.lm.special_token_id
seq_codes, _, _ = pattern.build_pattern_sequence(codes, special_id)
print(f"\nsequence_codes shape: {seq_codes.shape}")
print(f"special tokens по кодбукам: {[(seq_codes[0, k] == special_id).sum().item() for k in range(4)]}")

# Проверяем embedding весов на NaN
for k, emb in enumerate(model.lm.emb):
    nan_emb = emb.weight.isnan().any().item()
    print(f"emb[{k}] weights NaN: {nan_emb}, shape: {emb.weight.shape}, dtype: {emb.weight.dtype}")
