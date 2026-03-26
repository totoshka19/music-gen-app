# Music Generator

Локальный AI-генератор музыки на базе MusicGen Medium с LoRA fine-tuning.

## Стек

- **Модель**: `facebook/musicgen-medium` + LoRA (PEFT) — fine-tuning на FMA dataset
- **Backend**: FastAPI + Python 3.10, PyTorch 2.1.0 + CUDA 11.8
- **Frontend**: React 18 + TypeScript + Vite, WaveSurfer.js

## Структура

```
music-gen-app/
├── api/                  # FastAPI backend
│   ├── main.py           # REST API + mock-режим
│   ├── requirements.txt  # зависимости (с PyTorch/AudioCraft)
│   └── requirements_mock.txt  # зависимости для Replit (без GPU)
├── frontend/             # React + Vite (TypeScript)
│   └── src/
│       ├── hooks/        # useWaveSurfer, useGeneration, useTrackHistory
│       ├── utils/        # downloadTrack
│       ├── types.ts
│       ├── constants.ts
│       └── App.tsx
├── model/                # ML-скрипты
│   ├── prepare_dataset.py  # подготовка FMA dataset
│   ├── train.py            # LoRA fine-tuning
│   ├── test_model.py       # тест базовой модели
│   └── test_finetuned.py   # тест fine-tuned модели
├── .replit               # конфиг для деплоя на Replit
└── run_replit.sh         # скрипт запуска на Replit
```

## Запуск локально

**Требования**: Windows, NVIDIA GPU (≥8 GB VRAM), Python 3.10, Node.js 18+

```bash
# 1. Создать виртуальное окружение
python -m venv venv
venv\Scripts\activate

# 2. Установить зависимости backend
pip install -r api/requirements.txt

# 3. Запустить API (порт 8000)
cd api
uvicorn main:app --port 8000 --reload

# 4. В отдельном терминале — запустить frontend (порт 5173)
cd frontend
npm install
npm run dev
```

## API эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/api/v1/generate` | Поставить задачу в очередь |
| `GET` | `/api/v1/status/{task_id}` | Статус генерации |
| `GET` | `/api/v1/result/{task_id}` | Результат (URL аудио) |
| `GET` | `/api/v1/tracks` | История треков |
| `GET` | `/audio/{filename}` | Скачать аудио-файл |

### Пример запроса

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "lofi hip hop, relaxed piano, 85 bpm", "duration": 20}'
```

## Fine-tuning

Модель дообучена на [FMA (Free Music Archive)](https://github.com/mdeff/fma) dataset:
- 8 жанров: Hip-Hop, Electronic, Rock, Folk, Jazz, Classical, Soul-RnB, Pop
- 1046 аудио-сегментов по 10 секунд
- LoRA: `r=8`, `alpha=16`, 3 эпохи, LR=5e-5
- Обучение: RTX 4070 (12 GB VRAM), ~40 минут

```bash
# Подготовить датасет (нужна папка dataset/fma_small/)
python model/prepare_dataset.py

# Обучить
python model/train.py

# Протестировать результат
python model/test_finetuned.py
```

## Mock-режим (без GPU)

Для деплоя на Replit или тестирования без GPU:

```bash
MOCK=true uvicorn main:app --port 8000
```

В mock-режиме вместо реальной генерации возвращается синтетический аудио-файл (lo-fi аккорд с тремоло).

## Деплой на Replit

1. Загрузить проект на Replit
2. Установить зависимости: `pip install -r api/requirements_mock.txt && cd frontend && npm install && npm run build`
3. Запустить: `bash run_replit.sh`

Frontend собирается в `frontend/dist/` и раздаётся FastAPI через `StaticFiles`.
