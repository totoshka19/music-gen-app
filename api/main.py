import os
import time
import uuid
import numpy as np
import scipy.io.wavfile as wav_io
import torch
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ---- НАСТРОЙКИ ----
MODEL_NAME = 'facebook/musicgen-medium'
CHECKPOINT = '../checkpoints/epoch_3'
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
MOCK = os.getenv("MOCK", "false").lower() == "true"
SAMPLE_RATE = 32000
# -------------------

model = None

if not MOCK:
    from audiocraft.models import MusicGen
    from audiocraft.data.audio import audio_write
    from peft import PeftModel

    print("Загружаем модель...")
    model = MusicGen.get_pretrained(MODEL_NAME)
    if os.path.exists(CHECKPOINT):
        model.lm = PeftModel.from_pretrained(model.lm, CHECKPOINT)
        print("LoRA адаптер загружен")
    else:
        print("Чекпоинт не найден — используется базовая модель")
    model.lm.eval()
    print("Модель готова!")
else:
    print("MOCK-режим — модель не загружается (нет GPU)")


def _make_sample_wav() -> Path:
    """Генерирует демо-трек программно если его нет."""
    path = OUTPUT_DIR / "sample.wav"
    if path.exists():
        return path
    t = np.linspace(0, 20, 20 * SAMPLE_RATE, dtype=np.float32)
    # Простой lo-fi аккорд: A2 + E3 + A3 + C#4
    freqs = [110.0, 164.8, 220.0, 277.2, 329.6, 440.0]
    audio = sum(0.12 * np.sin(2 * np.pi * f * t) for f in freqs)
    # Добавляем лёгкий tremolo и лёгкий шум для «живости»
    audio *= (1 + 0.08 * np.sin(2 * np.pi * 4 * t))
    audio += 0.003 * np.random.randn(len(t)).astype(np.float32)
    audio = np.clip(audio * 0.7, -1.0, 1.0)
    wav_io.write(str(path), SAMPLE_RATE, (audio * 32767).astype(np.int16))
    return path


app = FastAPI(
    title="Music Generator API",
    description="REST API для генерации музыки локальной AI-моделью MusicGen Medium + LoRA fine-tuning.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory="outputs"), name="files")

tasks: dict = {}


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Описание музыки на английском",
                        example="lofi hip hop, relaxed piano, 85 bpm")
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
        "params": req.model_dump(),
    }
    background.add_task(do_generate, task_id, req)
    return {"task_id": task_id, "status": "queued", "created_at": tasks[task_id]["created_at"]}


@app.get("/api/v1/status/{task_id}", summary="Статус задачи")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Задача не найдена")
    t = tasks[task_id]
    result = {"task_id": task_id, "status": t["status"],
              "created_at": t["created_at"], "prompt": t["prompt"]}
    if t["status"] == "failed":
        result["error"] = t.get("error", "Неизвестная ошибка")
    return result


@app.get("/api/v1/result/{task_id}", summary="Получить результат")
async def get_result(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Задача не найдена")
    t = tasks[task_id]
    if t["status"] != "done":
        raise HTTPException(202, f"Ещё не готово. Статус: {t['status']}")
    return {
        "task_id": task_id,
        "status": "done",
        "file_url": f"/files/{task_id}.wav",
        "duration": t["params"]["duration"],
        "prompt": t["prompt"],
        "created_at": t["created_at"],
    }


@app.get("/api/v1/tracks", summary="История треков")
async def list_tracks():
    done = [
        {
            "task_id": k,
            "prompt": v["prompt"],
            "created_at": v["created_at"],
            "file_url": f"/files/{k}.wav",
            "duration": v["params"]["duration"],
        }
        for k, v in tasks.items()
        if v["status"] == "done"
    ]
    return {"tracks": done, "total": len(done)}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mode": "mock" if MOCK else "gpu",
        "device": "cpu" if MOCK else ("cuda" if torch.cuda.is_available() else "cpu"),
    }


def do_generate(task_id: str, req: GenerateRequest):
    tasks[task_id]["status"] = "generating"
    try:
        if MOCK:
            # Имитируем задержку генерации
            time.sleep(min(req.duration * 0.3, 8))
            sample = _make_sample_wav()
            import shutil
            shutil.copy(str(sample), str(OUTPUT_DIR / f"{task_id}.wav"))
        else:
            model.set_generation_params(
                duration=req.duration,
                temperature=req.temperature,
                cfg_coef=req.cfg_coef,
                top_k=250,
            )
            wav = model.generate([req.prompt])
            from audiocraft.data.audio import audio_write
            audio_write(
                str(OUTPUT_DIR / task_id),
                wav[0].cpu(),
                model.sample_rate,
                strategy="loudness",
            )
        tasks[task_id]["status"] = "done"
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        print(f"Ошибка генерации {task_id}: {e}")


# Раздаём собранный React-фронтенд (для Replit)
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="spa")
