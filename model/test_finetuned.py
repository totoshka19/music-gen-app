from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from peft import PeftModel
import torch

MODEL_NAME = 'facebook/musicgen-medium'
CHECKPOINT = 'checkpoints/epoch_10'

print("Загружаем базовую модель...")
model = MusicGen.get_pretrained(MODEL_NAME)
model.lm = PeftModel.from_pretrained(model.lm, CHECKPOINT)
model.lm.eval()
print("LoRA адаптер загружен")

model.set_generation_params(duration=20, cfg_coef=3.0, temperature=0.9)

prompts = [
    "lofi hip hop, relaxed piano, 85 bpm",
    "hip hop beat, heavy bass, energetic",
    "lo-fi chill, vinyl crackle, mellow",
]

for i, prompt in enumerate(prompts):
    print(f"Генерируем [{i + 1}/{len(prompts)}]: {prompt}")
    wav = model.generate([prompt])
    out = f"result_finetuned_{i}"
    audio_write(out, wav[0].cpu(), model.sample_rate, strategy="loudness")
    print(f"Сохранён: {out}.wav")

print("\nГотово! Сравните result_finetuned_*.wav с test_output.wav")
