# MusicGen — план реализации ТЗ

> **Контекст:** Windows, NVIDIA GeForce RTX 4070 (12 GB VRAM), всё локально и бесплатно.
> **ТЗ:** локальная модель + файнтюнинг → REST API → красивый веб-интерфейс. Опционально — интеграция с GPU-фермой.

---

## Структура проекта

```
music-gen-app/
├── model/
│   ├── test_model.py        # тест базовой генерации
│   ├── prepare_dataset.py   # подготовка датасета
│   ├── train.py             # скрипт файнтюнинга
│   └── test_finetuned.py    # тест после обучения
├── api/
│   ├── main.py              # FastAPI сервер
│   ├── outputs/             # папка с готовыми треками
│   └── requirements.txt
├── frontend/                # React + Vite
│   ├── src/
│   │   └── App.tsx
│   └── package.json
├── checkpoints/             # сохранённые LoRA-адаптеры
│   └── epoch_10/
├── dataset/
│   ├── processed/           # нарезанные .wav сегменты
│   └── metadata.jsonl       # описания к каждому сегменту
└── PLAN.md                  # этот файл
```

---

## Фаза 0 — Настройка окружения (1–2 дня)

### Шаг 1. Установить Python 3.10

- Скачать с https://python.org/downloads — **строго 3.10**, не 3.11/3.12 (AudioCraft несовместим)
- При установке поставить галочку **"Add Python to PATH"**
- Проверить:
```bash
python --version
# Должно вывести: Python 3.10.x
```

### Шаг 2. Установить Git

- Скачать с https://git-scm.com → установить с настройками по умолчанию
- Проверить: `git --version`

### Шаг 3. Установить CUDA 11.8

- Скачать с https://developer.nvidia.com/cuda-11-8-0-download-archive
- Выбрать: Windows → x86_64 → exe (local)
- Установить, перезагрузить компьютер
- Проверить: `nvcc --version` → должно показать release 11.8

### Шаг 4. Создать папку проекта и виртуальное окружение

```bash
mkdir music-gen-app
cd music-gen-app

python -m venv venv

# Активировать (Windows):
venv\Scripts\activate

# В терминале должно появиться (venv) в начале строки
```

> **Важно:** каждый раз при открытии нового терминала выполнять `venv\Scripts\activate`

### Шаг 5. Установить все зависимости

```bash
# PyTorch с поддержкой CUDA 11.8 (RTX 4070):
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# AudioCraft (MusicGen от Meta):
pip install audiocraft

# PEFT (для LoRA файнтюнинга):
pip install peft accelerate

# FastAPI (REST сервер):
pip install fastapi uvicorn python-multipart

# Для датасета:
pip install librosa soundfile pandas

# Прочее:
pip install python-dotenv requests
```

> Установка занимает 15–25 минут, PyTorch весит ~2.5 GB — это нормально.

---

## Фаза 1 — Запустить локальную модель (1–2 дня)

### Шаг 1. Выбор модели

RTX 4070 (12 GB VRAM) → использовать **`facebook/musicgen-medium`**

- `musicgen-small` — для 4–6 GB (хуже качество)
- `facebook/musicgen-medium` — **оптимально для 12 GB** ✓
- `musicgen-large` — для 16+ GB (при файнтюнинге вылетает по памяти на 12 GB)

### Шаг 2. Создать `model/test_model.py`

```python
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torch

MODEL_NAME = 'facebook/musicgen-medium'

print(f"Загружаем модель {MODEL_NAME}...")
model = MusicGen.get_pretrained(MODEL_NAME)
# Первый запуск скачает ~3.5 GB с Hugging Face — один раз

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используем: {device}")  # должно вывести: cuda

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
print("Готово! Файл: test_output.wav")
```

```bash
python model/test_model.py
```

После выполнения появится `test_output.wav` — прослушать. Это результат **до** файнтюнинга.

### Шаг 3. Протестировать 20–30 промптов (baseline)

Прогнать разные жанры и оценить по шкале 1–5, сравнивая с Suno.ai. Записать оценки — потом сравним с результатом после файнтюнинга.

### Шаг 4. Подобрать параметры генерации

```python
model.set_generation_params(
    duration=30,      # длина трека (10–120 секунд)
    temperature=0.9,  # случайность: 0.7 = предсказуемо, 1.2 = хаотично
    top_k=250,        # из скольких вариантов выбирает
    cfg_coef=3.0,     # строгость следования промпту: 1=свободно, 5=строго
)
# Поэкспериментировать с cfg_coef: 2.0 vs 4.0 — заметная разница
```

---

## Фаза 2 — Подготовить датасет (2–4 дня)

### Шаг 1. Определить целевой жанр

Выбрать 1–2 жанра для специализации. Рекомендация для MVP: **lofi hip-hop** или **ambient** — у них предсказуемая структура, файнтюнинг работает лучше всего.

### Шаг 2. Скачать датасет FMA (бесплатный)

- Сайт: https://github.com/mdeff/fma
- Скачать **fma_small.zip** (~7.2 GB, 8000 треков) — прямая ссылка:
  `https://os.unil.cloud.switch.ch/fma/fma_small.zip`
- Скачать **fma_metadata.zip** — там жанры и теги к каждому треку

```bash
# Распаковать в папку dataset/:
# Windows: правая кнопка → Извлечь в "dataset\"
```

### Шаг 3. Создать `model/prepare_dataset.py`

```python
import pandas as pd
import librosa
import soundfile as sf
import os
from pathlib import Path

TARGET_GENRE = 'Hip-Hop'  # менять под нужный жанр

tracks = pd.read_csv('dataset/metadata/tracks.csv', index_col=0, header=[0,1])
genre_col = ('track', 'genre_top')
selected = tracks[tracks[genre_col] == TARGET_GENRE].head(500)
print(f"Выбрано треков: {len(selected)}")

Path("dataset/processed").mkdir(parents=True, exist_ok=True)

def process_track(track_id):
    folder = str(track_id).zfill(6)[:3]
    src = f"dataset/fma_small/{folder}/{str(track_id).zfill(6)}.mp3"
    if not os.path.exists(src):
        return []
    try:
        audio, _ = librosa.load(src, sr=32000, mono=True)
        saved = []
        seg_len = 20 * 32000  # 20-секундные сегменты
        for i, start in enumerate(range(0, len(audio) - seg_len, seg_len)):
            seg = audio[start:start+seg_len]
            seg = seg / (max(abs(seg)) + 1e-8) * 0.9  # нормализация
            out = f"dataset/processed/{track_id}_{i:02d}.wav"
            sf.write(out, seg, 32000)
            saved.append(out)
        return saved
    except Exception as e:
        print(f"Ошибка {track_id}: {e}")
        return []

all_segments = []
for tid in selected.index:
    segs = process_track(tid)
    all_segments.extend(segs)
    if len(all_segments) % 100 == 0:
        print(f"Обработано сегментов: {len(all_segments)}")

print(f"Итого сегментов: {len(all_segments)}")
```

```bash
python model/prepare_dataset.py
```

### Шаг 4. Создать текстовые описания (`dataset/metadata.jsonl`)

```python
import json
import pandas as pd
from pathlib import Path

tracks = pd.read_csv('dataset/metadata/tracks.csv', index_col=0, header=[0,1])
TARGET_GENRE = 'Hip-Hop'

try:
    tags = pd.read_csv('dataset/metadata/echonest.csv', index_col=0, header=[0,1])
    has_echonest = True
except:
    has_echonest = False

processed_files = list(Path('dataset/processed').glob('*.wav'))

with open('dataset/metadata.jsonl', 'w') as f:
    for wav_path in processed_files:
        track_id = int(wav_path.stem.split('_')[0])
        genre = TARGET_GENRE.lower()
        caption = genre + " music"

        if has_echonest:
            try:
                tempo = tags.loc[track_id, ('audio_features', 'tempo')]
                energy = tags.loc[track_id, ('audio_features', 'energy')]
                mood = 'energetic' if energy > 0.6 else 'relaxed'
                caption = f"{genre}, {int(tempo)} bpm, {mood}"
            except:
                pass

        record = {"audio": wav_path.name, "caption": caption}
        f.write(json.dumps(record) + '\n')

print("dataset/metadata.jsonl создан")
```

> **Совет:** для лучшего качества — прослушать часть треков и дописать описания вручную.
> Формат: `"lofi hip hop, 85 bpm, mellow piano, soft kick, relaxed, vinyl crackle"`

---

## Фаза 3 — Fine-tuning (1–3 дня, 12–24 ч обучения)

### Что такое LoRA (кратко)

MusicGen Medium имеет 1.5 млрд параметров — менять все долго и дорого. LoRA добавляет небольшие "адаптеры" (~50 MB) поверх ключевых слоёв. Обучаются только они — в 10–20 раз быстрее и дешевле, качество почти такое же.

### Создать `model/train.py`

```python
from audiocraft.models import MusicGen
from peft import LoraConfig, get_peft_model
import torch, json
from torch.utils.data import Dataset, DataLoader
import torchaudio
from pathlib import Path

# ---- НАСТРОЙКИ ----
MODEL_NAME = 'facebook/musicgen-medium'
DATASET_DIR = 'dataset/processed'
METADATA_FILE = 'dataset/metadata.jsonl'
OUTPUT_DIR = 'checkpoints'
BATCH_SIZE = 1        # для 12 GB VRAM
GRAD_ACCUM = 8
LR = 1e-4
EPOCHS = 10
# -------------------

class AudioDataset(Dataset):
    def __init__(self):
        with open(METADATA_FILE) as f:
            self.items = [json.loads(l) for l in f if l.strip()]
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        item = self.items[idx]
        wav, sr = torchaudio.load(f"{DATASET_DIR}/{item['audio']}")
        if sr != 32000:
            wav = torchaudio.functional.resample(wav, sr, 32000)
        return {'wav': wav, 'caption': item['caption']}

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
print(f"Обучаемых параметров: {trainable:,} из {total:,} ({100*trainable/total:.1f}%)")

optimizer = torch.optim.AdamW(
    [p for p in model.lm.parameters() if p.requires_grad], lr=LR
)

dataset = AudioDataset()
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
Path(OUTPUT_DIR).mkdir(exist_ok=True)

for epoch in range(EPOCHS):
    model.lm.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        with torch.autocast(device):
            out = model.lm.compute_predictions(
                codes=model.compression_model.encode(batch['wav'].to(device))[0],
                conditions=[{'description': [c]} for c in batch['caption']],
            )
            loss = out.loss / GRAD_ACCUM

        loss.backward()
        total_loss += loss.item()

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.lm.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if step % 50 == 0:
            print(f"Эпоха {epoch+1}/{EPOCHS}, шаг {step}, loss={total_loss/(step+1):.4f}")

    save_path = f"{OUTPUT_DIR}/epoch_{epoch+1}"
    model.lm.save_pretrained(save_path)
    print(f"Чекпоинт сохранён: {save_path}")

print("Обучение завершено!")
```

```bash
# Запустить обучение (оставить на ночь):
python model/train.py

# В отдельном терминале следить за GPU:
nvidia-smi -l 5
```

**Что смотреть:**
- `loss` должен постепенно падать: 4–6 в начале → 2–3 после 500 шагов
- VRAM Usage 80–95% — нормально
- Если loss не падает 300+ шагов — уменьшить LR: `1e-4` → `5e-5`
- Если `CUDA out of memory` — уменьшить `r=8` → `r=4`

### Создать `model/test_finetuned.py`

```python
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from peft import PeftModel
import torch

MODEL_NAME = 'facebook/musicgen-medium'
CHECKPOINT = 'checkpoints/epoch_10'  # последний чекпоинт

base = MusicGen.get_pretrained(MODEL_NAME)
base.lm = PeftModel.from_pretrained(base.lm, CHECKPOINT)
base.lm.eval()

base.set_generation_params(duration=20, cfg_coef=3.0, temperature=0.9)

# Те же промпты что тестировали ДО обучения
prompts = [
    "lofi hip hop, relaxed piano, 85 bpm",
    "hip hop beat, heavy bass, energetic",
    "lo-fi chill, vinyl crackle, mellow",
]

for i, p in enumerate(prompts):
    wav = base.generate([p])
    audio_write(f"result_finetuned_{i}", wav[0].cpu(), base.sample_rate)
    print(f"Сохранён: result_finetuned_{i}.wav")
```

```bash
python model/test_finetuned.py
```

Сравнить `result_finetuned_*.wav` с `test_output_*.wav` — если звучит ближе к нужному стилю, файнтюнинг сработал.

---

## Фаза 4 — REST API на FastAPI (2–3 дня)

### Создать `api/main.py`

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uuid, os, torch
from pathlib import Path
from datetime import datetime
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from peft import PeftModel

# ---- НАСТРОЙКИ ----
MODEL_NAME = 'facebook/musicgen-medium'
CHECKPOINT = '../checkpoints/epoch_10'  # путь к LoRA адаптеру
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
# -------------------

print("Загружаем модель...")
model = MusicGen.get_pretrained(MODEL_NAME)
if os.path.exists(CHECKPOINT):
    model.lm = PeftModel.from_pretrained(model.lm, CHECKPOINT)
    print("LoRA адаптер загружен")
model.lm.eval()
print("Модель готова!")

app = FastAPI(
    title="MusicGen Local API",
    description="Генерация музыки локальной AI-моделью. Отправьте описание — получите .wav файл.",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.mount("/files", StaticFiles(directory="outputs"), name="files")

tasks: dict = {}

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Описание музыки на английском",
                        example="lofi hip hop, relaxed, piano, 85 bpm")
    duration: int = Field(default=30, ge=5, le=120,
                          description="Длина трека в секундах (5–120)")
    temperature: float = Field(default=0.9, ge=0.5, le=1.5)
    cfg_coef: float = Field(default=3.0, ge=1.0, le=6.0)

@app.post("/api/v1/generate", summary="Запустить генерацию трека")
async def generate(req: GenerateRequest, background: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "queued",
        "prompt": req.prompt,
        "created_at": datetime.now().isoformat(),
        "params": req.dict()
    }
    background.add_task(do_generate, task_id, req)
    return {"task_id": task_id, "status": "queued",
            "created_at": tasks[task_id]["created_at"]}

@app.get("/api/v1/status/{task_id}", summary="Статус задачи")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Задача не найдена")
    t = tasks[task_id]
    return {"task_id": task_id, "status": t["status"],
            "created_at": t["created_at"], "prompt": t["prompt"]}

@app.get("/api/v1/result/{task_id}", summary="Получить результат")
async def get_result(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Задача не найдена")
    t = tasks[task_id]
    if t["status"] != "done":
        raise HTTPException(202, f"Ещё не готово. Статус: {t['status']}")
    return {"task_id": task_id, "status": "done",
            "file_url": f"/files/{task_id}.wav",
            "duration": t["params"]["duration"],
            "prompt": t["prompt"]}

@app.get("/api/v1/tracks", summary="История треков")
async def list_tracks():
    done = [
        {"task_id": k, "prompt": v["prompt"],
         "created_at": v["created_at"], "file_url": f"/files/{k}.wav"}
        for k, v in tasks.items() if v["status"] == "done"
    ]
    return {"tracks": done, "total": len(done)}

def do_generate(task_id: str, req: GenerateRequest):
    tasks[task_id]["status"] = "generating"
    try:
        model.set_generation_params(
            duration=req.duration,
            temperature=req.temperature,
            cfg_coef=req.cfg_coef,
            top_k=250,
        )
        wav = model.generate([req.prompt])
        audio_write(str(OUTPUT_DIR / task_id),
                    wav[0].cpu(), model.sample_rate, strategy="loudness")
        tasks[task_id]["status"] = "done"
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        print(f"Ошибка генерации: {e}")
```

### Создать `api/requirements.txt`

```
fastapi
uvicorn
audiocraft
peft
torch==2.1.0
torchaudio==2.1.0
python-multipart
python-dotenv
```

### Запустить и проверить

```bash
cd api/
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Открыть в браузере:
- `http://localhost:8000/docs` — интерактивная документация (Swagger UI)
- `http://localhost:8000/redoc` — документация ReDoc

**Тест в браузере через /docs:**
1. POST `/api/v1/generate` → Try it out → ввести prompt → Execute
2. Скопировать `task_id` из ответа
3. GET `/api/v1/status/{task_id}` → вставить task_id → Execute
4. Когда `status: "done"` → GET `/api/v1/result/{task_id}` → открыть `file_url`

---

## Фаза 5 — Веб-интерфейс (3–4 дня)

### Установить Node.js

Скачать LTS с https://nodejs.org

### Создать проект

```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install axios wavesurfer.js
```

### Заменить `frontend/src/App.tsx`

```tsx
import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import WaveSurfer from 'wavesurfer.js'

const API = 'http://localhost:8000/api/v1'

type Status = 'idle' | 'queued' | 'generating' | 'done' | 'error'

interface Track {
  task_id: string
  prompt: string
  file_url: string
}

export default function App() {
  const [prompt, setPrompt] = useState('')
  const [duration, setDuration] = useState(20)
  const [status, setStatus] = useState<Status>('idle')
  const [tracks, setTracks] = useState<Track[]>([])
  const [currentUrl, setCurrentUrl] = useState('')
  const waveRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WaveSurfer | null>(null)

  useEffect(() => {
    if (!waveRef.current || !currentUrl) return
    wsRef.current?.destroy()
    wsRef.current = WaveSurfer.create({
      container: waveRef.current,
      waveColor: '#8B5CF6',
      progressColor: '#4C1D95',
      height: 64,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
    })
    wsRef.current.load(`http://localhost:8000${currentUrl}`)
  }, [currentUrl])

  const generate = async () => {
    if (!prompt.trim()) return
    setStatus('queued')
    try {
      const { data } = await axios.post(`${API}/generate`, { prompt, duration })
      const taskId = data.task_id

      const poll = setInterval(async () => {
        const { data: s } = await axios.get(`${API}/status/${taskId}`)
        setStatus(s.status)
        if (s.status === 'done') {
          clearInterval(poll)
          const { data: r } = await axios.get(`${API}/result/${taskId}`)
          setCurrentUrl(r.file_url)
          setTracks(prev => [{ task_id: taskId, prompt, file_url: r.file_url }, ...prev])
        }
        if (s.status === 'failed') { clearInterval(poll); setStatus('error') }
      }, 5000)
    } catch { setStatus('error') }
  }

  const labels: Record<Status, string> = {
    idle: 'Generate', queued: 'In queue...', generating: 'Generating...', done: 'Generate again', error: 'Error — retry'
  }

  return (
    <div style={{ minHeight: '100vh', background: '#0f0f1a', color: 'white', padding: '40px 24px', fontFamily: 'system-ui' }}>
      <div style={{ maxWidth: 680, margin: '0 auto' }}>

        <h1 style={{ fontSize: 28, fontWeight: 700, marginBottom: 6 }}>AI Music Generator</h1>
        <p style={{ color: '#888', marginBottom: 32, fontSize: 14 }}>Local MusicGen · fine-tuned · RTX 4070</p>

        <textarea
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          placeholder="Describe the music in English: lofi hip hop, relaxed piano, 85 bpm, vinyl crackle..."
          style={{ width: '100%', height: 100, background: '#1a1a2e', border: '1px solid #333', borderRadius: 10, padding: 14, color: 'white', fontSize: 14, resize: 'vertical', boxSizing: 'border-box' }}
        />

        <div style={{ display: 'flex', alignItems: 'center', gap: 12, margin: '12px 0' }}>
          <span style={{ color: '#888', fontSize: 13, whiteSpace: 'nowrap' }}>Duration: {duration}s</span>
          <input type="range" min={5} max={120} step={5} value={duration}
            onChange={e => setDuration(+e.target.value)} style={{ flex: 1 }} />
        </div>

        <button
          onClick={generate}
          disabled={['queued', 'generating'].includes(status)}
          style={{ width: '100%', padding: 14, background: status === 'error' ? '#7f1d1d' : '#7c3aed', border: 'none', borderRadius: 10, color: 'white', fontSize: 15, fontWeight: 600, cursor: 'pointer' }}>
          {labels[status]}
        </button>

        {currentUrl && (
          <div style={{ background: '#1a1a2e', borderRadius: 12, padding: 16, marginTop: 24 }}>
            <div ref={waveRef} style={{ marginBottom: 12 }} />
            <div style={{ display: 'flex', gap: 10 }}>
              <button onClick={() => wsRef.current?.playPause()}
                style={{ flex: 1, padding: 10, background: '#7c3aed', border: 'none', borderRadius: 8, color: 'white', cursor: 'pointer' }}>
                Play / Pause
              </button>
              <a href={`http://localhost:8000${currentUrl}`} download
                style={{ flex: 1, padding: 10, background: '#1e1e3a', border: '1px solid #444', borderRadius: 8, color: '#a78bfa', textAlign: 'center', textDecoration: 'none', fontSize: 14, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                Download .wav
              </a>
            </div>
          </div>
        )}

        {tracks.length > 0 && (
          <div style={{ marginTop: 32 }}>
            <h2 style={{ fontSize: 15, fontWeight: 600, marginBottom: 12, color: '#ccc' }}>History</h2>
            {tracks.map(t => (
              <div key={t.task_id} onClick={() => setCurrentUrl(t.file_url)}
                style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#1a1a2e', borderRadius: 8, padding: '10px 14px', marginBottom: 6, cursor: 'pointer' }}>
                <span style={{ fontSize: 13, color: '#ddd', flex: 1 }}>{t.prompt}</span>
                <span style={{ fontSize: 11, color: '#7c3aed', marginLeft: 12 }}>▶ play</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
```

### Запустить фронтенд

```bash
cd frontend/
npm run dev
# Открыть: http://localhost:5173
```

---

## Запуск всей системы

Открыть **два терминала** одновременно:

```bash
# Терминал 1 — API (активировать venv!):
cd music-gen-app/
venv\Scripts\activate
cd api/
uvicorn main:app --host 0.0.0.0 --port 8000

# Терминал 2 — Фронтенд:
cd music-gen-app/frontend/
npm run dev
```

Открыть `http://localhost:5173` → вводить промпты → слушать треки.

Документация API: `http://localhost:8000/docs`

---

## Фаза 6 — GPU-ферма через Kaggle (опционально, бонус по ТЗ)

> ТЗ говорит: "если добавишь — супер". Kaggle даёт 30 часов GPU в неделю бесплатно.

### Шаг 1. Настроить Kaggle

- Зарегистрироваться на kaggle.com → верифицировать телефон (нужно для GPU)
- Создать новый Notebook → Settings → Accelerator → GPU T4 x2
- Загрузить LoRA-чекпоинт как Kaggle Dataset: Account → Datasets → New Dataset → папка `checkpoints/`

### Шаг 2. Код ноутбука на Kaggle

```python
# Вставить в ячейки ноутбука на Kaggle:
!pip install audiocraft peft -q

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from peft import PeftModel
import os

model = MusicGen.get_pretrained('facebook/musicgen-medium')
model.lm = PeftModel.from_pretrained(model.lm, '/kaggle/input/ваш-датасет/epoch_10')
model.lm.eval()

from flask import Flask, request, jsonify, send_file
from threading import Thread
app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    model.set_generation_params(duration=data.get('duration', 20), cfg_coef=3.0)
    wav = model.generate([data['prompt']])
    audio_write('/tmp/output', wav[0].cpu(), model.sample_rate)
    return send_file('/tmp/output.wav', mimetype='audio/wav')

# Бесплатный публичный URL через ngrok:
!pip install pyngrok -q
from pyngrok import ngrok
ngrok.set_auth_token("ТОКЕН_С_NGROK_COM")  # бесплатный аккаунт на ngrok.com
tunnel = ngrok.connect(5000)
print("URL фермы:", tunnel.public_url)  # сохранить этот URL

Thread(target=lambda: app.run(port=5000)).start()
```

### Шаг 3. Подключить к API

В `api/main.py` в функции `do_generate` добавить переключение:

```python
import os, requests as req_lib

FARM_URL = os.getenv("KAGGLE_FARM_URL", "")  # URL из ngrok, если запущен

def do_generate(task_id: str, req: GenerateRequest):
    tasks[task_id]["status"] = "generating"
    try:
        if FARM_URL:
            # Генерация на Kaggle GPU
            resp = req_lib.post(f"{FARM_URL}/generate",
                json={"prompt": req.prompt, "duration": req.duration},
                timeout=300)
            with open(str(OUTPUT_DIR / f"{task_id}.wav"), 'wb') as f:
                f.write(resp.content)
        else:
            # Локальная генерация
            model.set_generation_params(duration=req.duration, cfg_coef=req.cfg_coef)
            wav = model.generate([req.prompt])
            audio_write(str(OUTPUT_DIR / task_id), wav[0].cpu(), model.sample_rate)
        tasks[task_id]["status"] = "done"
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
```

Запустить API с переменной окружения:
```bash
set KAGGLE_FARM_URL=https://xxxx.ngrok-free.app
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Итоговый чеклист

- [ ] Python 3.10 установлен, `venv` активен
- [ ] CUDA 11.8 установлен, `nvcc --version` показывает 11.8
- [ ] Все зависимости установлены через pip
- [ ] `test_output.wav` сгенерирован — базовая модель работает
- [ ] Датасет FMA скачан и распакован в `dataset/`
- [ ] `dataset/metadata.jsonl` создан (аудио + описания)
- [ ] `train.py` запущен, loss падает, чекпоинты сохраняются
- [ ] `result_finetuned_*.wav` звучат лучше базовых
- [ ] API запущен на `localhost:8000`, `/docs` открывается
- [ ] Фронтенд запущен на `localhost:5173`, генерация работает
- [ ] (опционально) Kaggle ноутбук работает, URL фермы подключён к API

---

## Справочник: команды быстрого запуска

```bash
# Активировать окружение (каждый раз):
cd music-gen-app/
venv\Scripts\activate

# Тест модели:
python model/test_model.py

# Обучение:
python model/train.py

# Тест после обучения:
python model/test_finetuned.py

# Запуск API:
cd api/ && uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Запуск фронтенда:
cd frontend/ && npm run dev
```

---

*RTX 4070 · 12 GB VRAM · Windows · MusicGen Medium · LoRA r=8 · FastAPI · React + Vite*
