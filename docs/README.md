# Инструкция по запуску

Проект поддерживает два основных режима:

* **локальный запуск** — уже полностью подготовлен в папке `local-run`
* **удалённый запуск** — конфиг job загружается по URL

Также trainer можно собирать не только на дефолтном базовом образе, но и **на основе другого образа с другой моделью**.

---

# 1. Требования

Перед запуском должно быть установлено: 

* Docker
* поддержка GPU в Docker
* NVIDIA Container Toolkit, если используется NVIDIA GPU

Проверка, что Docker видит GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Если команда не работает, сначала нужно настроить доступ Docker к GPU.

---

# 2. Сборка образа

## Обычная сборка

Из корня проекта:

```bash
docker build -f docker/Dockerfile -t igortet/itk-executor-trainer:latest .
```

---

## Сборка на базе другого образа с другой моделью

`docker/Dockerfile` поддерживает базовый образ через аргумент:

```dockerfile
ARG BASE_IMAGE=igortet/model-qwen-7b
FROM ${BASE_IMAGE}
```

Это значит, что trainer можно собрать поверх любого другого образа, в котором уже лежит нужная модель.

### Пример сборки с другим базовым образом

```bash
docker build \
  --build-arg BASE_IMAGE=my-registry/my-model-image:latest \
  -f docker/Dockerfile \
  -t igortet/itk-executor-trainer:latest .
```

Например:

```bash
docker build \
  --build-arg BASE_IMAGE=igortet/model-llama-3-8b \
  -f docker/Dockerfile \
  -t igortet/itk-executor-trainer:llama3-8b .
```

---

## Что важно при сборке с другим model-image

Нужно, чтобы в базовом образе:

* модель уже была внутри контейнера
* был известен путь к модели
* этот путь совпадал с тем, что указано в конфиге

Например, если в конфиге указано:

```json
"model": {
  "source": "local",
  "local_path": "/app"
}
```

то trainer ожидает модель внутри контейнера по пути:

```bash
/app
```

### Если в новом базовом образе модель тоже лежит в `/app`

Менять конфиг не нужно. Достаточно пересобрать trainer с новым `BASE_IMAGE`.

### Если модель лежит в другом месте

Тогда нужно обновить путь в конфиге, например:

```json
"model": {
  "source": "local",
  "local_path": "/models/llama3"
}
```

---

# 3. Локальный запуск

Для локального режима **всё уже подготовлено в папке `local-run`**.

Внутри уже есть:

* `run.sh`
* `job.local.json`
* `data/`
* `output/`

## Как запускать

Перейти в папку:

```bash
cd local-run
```

Запустить:

```bash
chmod +x run.sh
./run.sh
```

Этого достаточно.

---

## Что делает `run.sh`

Скрипт сам:

* создаёт папку `output`, если её нет
* запускает контейнер
* пробрасывает GPU
* монтирует нужные файлы в контейнер

Используются такие маппинги:

* `./data` → `/data`
* `./job.local.json` → `/configs/job.local.json`
* `./output` → `/output`

И передаются переменные окружения:

```bash
CONFIG_SOURCE=local
CONFIG_REF=/configs/job.local.json
```

---

## Что нужно проверить перед локальным запуском

Нужно убедиться, что:

* Docker видит GPU
* образ `igortet/itk-executor-trainer:latest` уже собран
* в `local-run/data` лежат нужные файлы датасета
* в `local-run/job.local.json` корректно указан путь к модели внутри контейнера

---

## Если образ ещё не собран

Из корня проекта:

```bash
docker build -f docker/Dockerfile -t igortet/itk-executor-trainer:latest .
```

Потом:

```bash
cd local-run
./run.sh
```

---

## Ручной запуск без `run.sh`

Если нужно запустить контейнер вручную:

```bash
cd local-run

docker run -it --rm \
  --gpus all \
  --shm-size 16g \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/job.local.json:/configs/job.local.json" \
  -v "$(pwd)/output:/output" \
  -e CONFIG_SOURCE=local \
  -e CONFIG_REF=/configs/job.local.json \
  igortet/itk-executor-trainer:latest
```

---

## Какой конфиг используется в local-run

Локально используется файл:

```bash
local-run/job.local.json
```

Он монтируется в контейнер как:

```bash
/configs/job.local.json
```

В этом конфиге уже настроены:

* модель
* train dataset
* evaluation dataset
* пути к output
* параметры обучения
* параметры evaluation

---

# 4. Удалённый запуск

Удалённый режим нужен, когда конфиг job находится по URL, а статусы и артефакты могут отправляться во внешний backend.

---

## Рекомендуемый способ: через `JOB_CONFIG_URL`

Это основной и предпочтительный вариант.

Пример запуска:

```bash
docker run --rm -it \
  --gpus all \
  --shm-size 16g \
  -v "$(pwd)/output:/output" \
  -e JOB_CONFIG_URL=http://YOUR_HOST:8010/files/job.remote.json \
  -e HF_TOKEN=hf_xxx \
  igortet/itk-executor-trainer:latest
```

---

## Альтернативный способ: через `CONFIG_SOURCE=remote`

После исправленного entrypoint можно запускать и так:

```bash
docker run --rm -it \
  --gpus all \
  --shm-size 16g \
  -v "$(pwd)/output:/output" \
  -e CONFIG_SOURCE=remote \
  -e CONFIG_REF=http://YOUR_HOST:8010/files/job.remote.json \
  -e HF_TOKEN=hf_xxx \
  igortet/itk-executor-trainer:latest
```

Но предпочтительный вариант всё равно — через `JOB_CONFIG_URL`.

---

## Что обычно есть в удалённом конфиге

В remote-конфиге обычно задаются:

* модель
* train dataset по URL
* eval dataset по URL
* callback URL для статусов и прогресса
* upload endpoints
* auth tokens или bearer token

---

## Что даёт remote bootstrap

Если запуск идёт через `--job-config-url`, сервис использует специальную загрузку remote config и может подхватывать не только сам job config, но и дополнительные метаданные:

* `logs_url`
* `status_url`
* `progress_url`
* `final_url`
* `callback_auth_token`

Это нужно для корректной отправки:

* статусов
* прогресса
* финального результата
* потоковых логов

---

# 5. Удалённый запуск через docker compose

Для remote-режима в проекте уже есть:

* `docker-compose.remote.yml`
* `.env.remote.example`

## Запуск

Из корня проекта:

```bash
docker compose -f docker-compose.remote.yml up --build
```

---

## Что использует compose

`docker-compose.remote.yml`:

* запускает контейнер с GPU
* монтирует `./output:/output`
* использует переменные из `.env.remote.example`

Пример `.env.remote.example`:

```env
CONFIG_SOURCE=remote
CONFIG_REF=http://192.168.31.32:8010/files/job.remote.json
HF_TOKEN=hf_xxx
```

При желании можно заменить `CONFIG_REF` на свой URL или использовать `JOB_CONFIG_URL` через `environment`.

---

# 6. Куда складываются результаты

Во всех режимах результаты пишутся в каталог `output`, а внутри создаётся отдельная папка запуска с timestamp.

Обычно внутри будут:

* `logs/trainer.log`
* `logs/effective-job.json`
* `job-result.json`
* `lora/...`
* `merged/...`
* `metrics/...`
* `evaluation/...`
* `downloads/...`

---

## Локальный режим

Результаты лежат в:

```bash
local-run/output
```

---

## Удалённый режим

Если запуск идёт из корня проекта с:

```bash
-v "$(pwd)/output:/output"
```

то результаты окажутся в:

```bash
output
```

---

# 7. Когда нужен `HF_TOKEN`

`HF_TOKEN` нужен, если:

* включён upload в Hugging Face
* включён publish в Hugging Face
* нужен доступ к приватным HF-репозиториям

Если Hugging Face не используется, токен можно не передавать.

---

# 8. Проверка и типичные проблемы

## Проверить, что есть образ

```bash
docker images | grep itk-executor-trainer
```

---

## Проверить локальные данные

```bash
ls -la local-run/data
```

---

## Проверить GPU из Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## Если не стартует локальный запуск

Проверь:

* собран ли образ
* доступен ли GPU
* есть ли файлы в `local-run/data`
* совпадает ли `model.local_path` в `local-run/job.local.json` с путём модели внутри базового образа

---

## Если не стартует удалённый запуск

Проверь:

* доступен ли `JOB_CONFIG_URL` из контейнера
* отдаёт ли backend валидный JSON
* корректны ли callback URL
* передан ли `HF_TOKEN`, если нужен HF

---

## Если evaluation падает по памяти

Уменьши в конфиге:

* `evaluation.batch_size`
* `evaluation.max_num_seqs`
* `evaluation.max_num_batched_tokens`
* `evaluation.max_model_len`
* `evaluation.gpu_memory_utilization`

---

# 9. Быстрые команды

## Локальный запуск

```bash
cd local-run
./run.sh
```

---

## Пересобрать trainer

```bash
docker build -f docker/Dockerfile -t igortet/itk-executor-trainer:latest .
```

---

## Собрать trainer на другом базовом model-image

```bash
docker build \
  --build-arg BASE_IMAGE=my-registry/my-model-image:latest \
  -f docker/Dockerfile \
  -t igortet/itk-executor-trainer:latest .
```

---

## Удалённый запуск

```bash
docker run --rm -it \
  --gpus all \
  --shm-size 16g \
  -v "$(pwd)/output:/output" \
  -e JOB_CONFIG_URL=http://YOUR_HOST:8010/files/job.remote.json \
  -e HF_TOKEN=hf_xxx \
  igortet/itk-executor-trainer:latest
```

---

## Удалённый запуск через compose

```bash
docker compose -f docker-compose.remote.yml up --build
```