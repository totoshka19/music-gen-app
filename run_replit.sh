#!/bin/bash
set -e

echo "=== Установка Python-зависимостей ==="
pip3 install -r api/requirements_mock.txt --quiet

echo "=== Сборка фронтенда ==="
cd frontend
npm install --silent
npm run build
cd ..

echo "=== Запуск API ==="
cd api
MOCK=true uvicorn main:app --host 0.0.0.0 --port 8000
