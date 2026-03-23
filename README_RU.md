# LLM Trainer Service

Надежный Docker-сервис для дообучения (fine-tuning) LLM на базе GPU-машин с использованием Unsloth (LoRA/QLoRA) и vLLM для оценки.

## 1. Обзор
Сервис предназначен для автоматизации процесса fine-tuning современных LLM. Он поддерживает эффективные методы дообучения (LoRA, QLoRA) и обеспечивает высокую надежность выполнения за счет системы повторных попыток, контроля ресурсов и изолированных сред инференса.

## 2. Быстрый старт

### Локальный запуск
1. Подготовьте конфиг `job-config.json` и данные.
2. Запустите Docker:
```bash
docker run --gpus all \
  -v $(pwd)/data:/data \
  -v $(pwd)/output:/output \
  -v $(pwd)/job-config.json:/config/job-config.json \
  trainer-service --config /config/job-config.json
```

### Удаленный запуск (Remote Executor)
Передайте URL конфигурации через переменную окружения или аргумент:
```bash
docker run --gpus all \
  -e JOB_CONFIG_URL="https://api.example.com/jobs/123/config" \
  trainer-service --job-config-url "https://api.example.com/jobs/123/config"
```

## 3. Требования
- **Docker** + **NVIDIA Container Toolkit**.
- **GPU**: NVIDIA (рекомендуется 24GB+ VRAM для моделей 7B+).
- **Диск**: 50GB+ свободного места в `/output`.

## 4. Конфигурация (config.json)

### Основные поля
- `job_name` (string): Имя задания, используется в именах файлов.
- `seed` (int): Random seed для воспроизводимости.

### Модель (`model`)
- `source`: `"local"` или `"huggingface"`.
- `repo_id`: ID модели на HF (например, `Qwen/Qwen2.5-7B`).
- `local_path`: Путь к локальной модели.
- `revision`: Ветка/commit на HF.
- `trust_remote_code`: Разрешить выполнение кода из репозитория.

### Датасет (`dataset`)
- `source`: `"local"` или `"url"`.
- `train_path`/`train_url`: Путь или URL обучающих данных (JSON/JSONL).
- `format`: `"instruction_output"`, `"messages"`, `"prompt_completion"`.

### Обучение (`training`)
- `method`: `"lora"` или `"qlora"`.
- `per_device_train_batch_size`: Размер батча на устройство.
- `num_train_epochs`: Количество эпох.
- `learning_rate`: Скорость обучения.
- `max_grad_norm`: Максимальная норма градиента.

### Оценка (`evaluation`)
- `enabled`: Включить/выключить оценку.
- `engine`: `"vllm"`.
- `retry_tries`: Количество попыток при сбое (с авто-уменьшением параметров памяти).

## 5. Режимы работы
- **Local mode**: Все данные и результаты остаются на локальной машине. Идеально для приватных экспериментов.
- **Remote mode**: Сервис активно взаимодействует с внешним бэкендом: стримит логи, отправляет статусы и выгружает артефакты.

## 6. Отчетность и Callbacks
Сервис отправляет POST-запросы на указанные URL со следующим полезным грузом:

**Пример статуса:**
```json
{
  "job_id": "123",
  "event": "status",
  "status": "running",
  "stage": "training",
  "progress": 45.5,
  "message": "Step 500/1000"
}
```
**Логи:** Стримятся чанками с указанием `offset`.

## 7. Hugging Face
Для публикации установите `HF_TOKEN`.
- `push_lora`: Выгрузить адаптеры.
- `push_merged`: Выгрузить объединенную 16-bit модель.
- `private`: Создать приватный репозиторий.

## 8. FAQ и Troubleshooting
- **В: Ошибка "Out of Memory" при обучении?**
  - О: Уменьшите `per_device_train_batch_size` или используйте `qlora`.
- **В: Как отключить выгрузку на HF?**
  - О: Установите `"huggingface": {"enabled": false}` в конфиге.
- **В: Можно ли запустить без интернета?**
  - О: Да, при условии, что `model.source="local"` и `dataset.source="local"`.

## 9. Артефакты (в /output)
- `job-result.json`: Финальный отчет.
- `trainer.log`: Журнал событий.
- `lora/`, `merged/`: Результаты обучения.
- `evaluation/`: Метрики и детали оценки.

---
Разработано с акцентом на стабильность и удобство использования в production.
