# Music Generator — Frontend

React 18 + TypeScript + Vite.

## Запуск

```bash
npm install
npm run dev     # dev-сервер на порту 5173
npm run build   # production сборка в dist/
```

## Структура

```
src/
├── hooks/
│   ├── useWaveSurfer.ts    # WaveSurfer lifecycle, loadTrack, playPause
│   ├── useGeneration.ts    # polling статуса задачи, таймер
│   └── useTrackHistory.ts  # история треков из API
├── utils/
│   └── downloadTrack.ts    # скачивание аудио через blob
├── types.ts                # Status, Track
├── constants.ts            # BASE_URL, API, STATUS_LABELS
├── App.tsx
└── App.css
```

API по умолчанию: `http://localhost:8000`
