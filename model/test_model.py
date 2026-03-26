from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torch

MODEL_NAME = 'facebook/musicgen-medium'

print(f"Загружаем модель {MODEL_NAME}...")
print("Первый запуск скачает ~3.5 GB с Hugging Face — это нормально, один раз.")
model = MusicGen.get_pretrained(MODEL_NAME)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используем: {device}")

model.set_generation_params(
    duration=10,
    temperature=0.9,
    top_k=250,
    cfg_coef=3.0,
)

prompt = "lofi hip hop, relaxed piano, soft drums, 85 bpm"
print(f"Генерируем: {prompt}")

wav = model.generate([prompt])
audio_write("test_output", wav[0].cpu(), model.sample_rate, strategy="loudness")
print("Готово! Файл сохранён: test_output.wav")
