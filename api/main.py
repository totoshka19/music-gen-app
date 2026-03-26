from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uuid
import os
import torch
from pathlib import Path
from datetime import datetime
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from peft import PeftModel

# ---- НАСТРОЙКИ ----
MODEL_NAME = 'facebook/musicgen-medium'
CHECKPOINT = '../checkpoints/epoch_10'
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
# -------------------

print("Загружаем модель...")
model = MusicGen.get_pretrained(MODEL_NAME)
if os.path.exists(CHECKPOINT):
    model.lm = PeftModel.from_pretrained(model.lm, CHECKPOINT)
    print("LoRA адаптер загружен")
else:
    print("Чекпоинт не найден — используется базовая модель")
model.lm.eval()
print("Модель готова!")

app = FastAPI(
    title="MusicGen Local API",
    description="Генерация музыки локальной AI-моделью. Отправьте описание — получите .wav файл.",
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
    result = {"task_id": task_id, "status": t["status"], "created_at": t["created_at"], "prompt": t["prompt"]}
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
    return {"status": "ok", "device": "cuda" if torch.cuda.is_available() else "cpu"}


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
