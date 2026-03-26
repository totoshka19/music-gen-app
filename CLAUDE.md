# CLAUDE.md — инструкции для Claude Code

## Контекст проекта

Тестовое задание для Lofty. Локальный AI-генератор музыки:
MusicGen Medium → LoRA fine-tuning → FastAPI → React UI.
Всё работает локально на Windows, RTX 4070 (12 GB VRAM).

## Структура проекта

```
music-gen-app/
├── model/       # ML-скрипты (тест, датасет, обучение)
├── api/         # FastAPI бекенд
│   └── outputs/ # сгенерированные треки
├── frontend/    # React + Vite (TypeScript)
├── checkpoints/ # LoRA-адаптеры после обучения
├── dataset/     # аудио-сегменты и metadata.jsonl
└── venv/        # виртуальное окружение Python 3.10
```

## Запуск проекта

```bash
# Активировать окружение (каждый раз):
venv\Scripts\activate

# API (порт 8000):
cd api && uvicorn main:app --port 8000 --reload

# Frontend (порт 5173):
cd frontend && npm run dev
```

## Ограничения окружения

- Python строго **3.10** — AudioCraft несовместим с 3.11+
- PyTorch **2.1.0** + CUDA **11.8** — не менять версии
- `BATCH_SIZE=1`, LoRA `r=8` — лимит VRAM 12 GB (RTX 4070)
- Модель: `facebook/musicgen-medium` — не предлагать `large`

## Стиль кода

- Python: типизация там, где не очевидно; без лишних комментариев
- React/TypeScript: только функциональные компоненты
- Не добавлять новые зависимости без явной просьбы

## Запрещено

- Не коммитить: `venv/`, `checkpoints/`, `dataset/`, `api/outputs/`
- Не предлагать Docker/Kubernetes — проект локальный
- Не менять порты: API=8000, Frontend=5173
- Не обновлять версии PyTorch/AudioCraft без явного запроса

## Дизайн-система фронтенда

Стиль: современный тёмный, вдохновлён Spotify + Linear. Никакого "generic AI" — чистый, профессиональный UI.

### Цветовая палитра

```
Фон страницы:   #08080f   (глубокий тёмно-синий, не чистый чёрный)
Поверхность:    #111120   (карточки, панели)
Поверхность 2:  #1c1c2e   (вложенные элементы, инпуты)
Граница:        #2e2e45   (subtle borders)

Акцент:         #6366f1   (indigo — основной цвет действий)
Акцент hover:   #4f46e5
Акцент свечение: rgba(99, 102, 241, 0.15)

Cyan-акцент:    #22d3ee   (для визуализации звука, прогресса)

Текст основной: #f1f5f9
Текст вторичный:#94a3b8
Текст disabled: #4a5568

Успех:          #10b981
Ошибка:         #ef4444
```

### Типографика

- Шрифт: `Inter` (подключить через Google Fonts) или `system-ui` как fallback
- Заголовок h1: 28px, weight 700, letter-spacing -0.5px
- Подзаголовки: 13px, weight 500, color #94a3b8, uppercase, letter-spacing 1px
- Основной текст: 14px, weight 400, line-height 1.6

### Компоненты

**Кнопки:**
- Primary: bg `#6366f1`, radius 10px, padding 12px 24px, font-weight 600
- На hover: bg `#4f46e5` + box-shadow `0 0 20px rgba(99,102,241,0.3)`
- Disabled: opacity 0.5, cursor not-allowed

**Инпуты / Textarea:**
- bg `#1c1c2e`, border `1px solid #2e2e45`, radius 10px
- На focus: border-color `#6366f1`, box-shadow `0 0 0 3px rgba(99,102,241,0.1)`
- Placeholder color: `#4a5568`

**Карточки:**
- bg `#111120`, border `1px solid #2e2e45`, radius 14px
- На hover: border-color `#6366f1`, transition 200ms

**Waveform (визуализация звука):**
- waveColor: `#6366f1`
- progressColor: `#22d3ee`
- Высота 72px, barWidth 2, barGap 1, barRadius 3

### Правила

- Все переходы: `transition: all 0.2s ease`
- Скругления: 10px для элементов, 14px для карточек, 50px для тегов/бейджей
- Никаких резких белых фонов
- Spacing кратен 4px (8, 12, 16, 24, 32, 48)
- Максимальная ширина контента: 720px, по центру

## Язык коммитов

Все git-коммиты писать **на русском языке**.

Формат:
```
<тип>: <краткое описание на русском>

<опциональное тело — если нужно пояснение>
```

Примеры типов: `feat`, `fix`, `refactor`, `docs`, `chore`, `test`

Примеры сообщений:
- `feat: добавить FastAPI эндпоинт генерации музыки`
- `fix: исправить ошибку загрузки LoRA-адаптера`
- `chore: настроить виртуальное окружение Python 3.10`
