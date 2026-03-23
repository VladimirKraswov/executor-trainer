# Конфигурация `local-run/job.local.json`

Файл `local-run/job.local.json` — это **не одиночный job**, а **bundle-конфиг**:

* `defaults` — общие настройки по умолчанию
* `jobs` — список конкретных запусков, которые наследуют `defaults` и при необходимости переопределяют отдельные поля

Это позволяет один раз описать общие параметры модели, обучения и evaluation, а в `jobs` менять только то, что отличается.

---

## Как устроен файл

Упрощённо структура такая:

```json
{
  "defaults": {
    "...": "общие настройки"
  },
  "jobs": [
    {
      "...": "настройки конкретного запуска"
    }
  ]
}
```

Во время старта сервис делает следующее:

1. берёт `defaults`
2. накладывает поверх них настройки из конкретного job
3. получает итоговый job config
4. запускает pipeline

---

# Пример из `local-run/job.local.json`

В твоём случае используется один job:

```json
{
  "defaults": {
    "...": "общие настройки"
  },
  "jobs": [
    {
      "job_id": "local_run",
      "job_name": "local-run",
      "dataset": {
        "source": "local",
        "train_path": "/data/train.jsonl",
        "format": "instruction_output",
        "input_field": "input",
        "output_field": "output"
      },
      "evaluation": {
        "enabled": true,
        "engine": "vllm",
        "target": "merged",
        "batch_size": 32,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 8192,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "max_new_tokens": 8,
        "temperature": 0.0,
        "do_sample": false,
        "score_min": 0.0,
        "score_max": 5.0,
        "parsing_regex": "^\\s*([0-5](?:\\.0)?)\\s*$",
        "dataset": {
          "source": "local",
          "path": "/data/eval.jsonl",
          "format": "jsonl",
          "task": "score_prediction",
          "prompt_field": "input",
          "score_field": "output",
          "tags_field": "details.hash"
        }
      }
    }
  ]
}
```

---

# Главное правило по путям

Все пути в конфиге — это **пути внутри контейнера**, а не на твоём компьютере.

Например:

```json
"train_path": "/data/train.jsonl"
```

означает, что файл должен быть доступен внутри контейнера по пути `/data/train.jsonl`.

Это работает потому, что `run.sh` монтирует:

```bash
./data -> /data
```

---

# Разбор `defaults`

## `mode`

```json
"mode": "local"
```

Режим запуска job.
Для `local-run` должен быть `local`.

---

## `model`

Пример:

```json
"model": {
  "source": "local",
  "local_path": "/app",
  "trust_remote_code": false,
  "load_in_4bit": true,
  "dtype": "bfloat16",
  "max_seq_length": 256
}
```

### Что означает

* `source: "local"` — модель берётся из файловой системы контейнера
* `local_path` — где внутри контейнера лежит модель
* `trust_remote_code` — разрешать ли кастомный код модели
* `load_in_4bit` — загружать ли модель в 4-битном виде
* `dtype` — тип вычислений
* `max_seq_length` — максимальная длина последовательности модели

### Когда менять

Менять это нужно, если:

* ты собрал trainer на базе другого model-image
* у новой модели другой путь
* нужна другая точность
* нужен другой context length

### Пример для другой модели

Если в базовом образе модель лежит в `/models/llama3`, то:

```json
"model": {
  "source": "local",
  "local_path": "/models/llama3",
  "trust_remote_code": false,
  "load_in_4bit": true,
  "dtype": "bfloat16",
  "max_seq_length": 4096
}
```

---

## `training`

Пример:

```json
"training": {
  "method": "qlora",
  "max_seq_length": 256,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 1,
  "num_train_epochs": 1,
  "learning_rate": 0.0001,
  "warmup_ratio": 0.03,
  "logging_steps": 1,
  "save_steps": 10,
  "eval_steps": 10,
  "bf16": true,
  "packing": false,
  "save_total_limit": 1,
  "optim": "adamw_8bit"
}
```

### Что означает

* `method` — способ fine-tuning, обычно `lora` или `qlora`
* `max_seq_length` — длина sequence для train
* `per_device_train_batch_size` — batch size на устройство
* `gradient_accumulation_steps` — накопление градиентов
* `num_train_epochs` — число эпох
* `learning_rate` — learning rate
* `warmup_ratio` — доля warmup
* `logging_steps` — как часто логировать
* `save_steps` — как часто сохранять checkpoint
* `eval_steps` — шаг evaluation внутри trainer, если есть validation
* `bf16` — использовать bfloat16
* `packing` — упаковывать несколько коротких примеров в один sequence
* `save_total_limit` — сколько checkpoint хранить
* `optim` — оптимизатор

### Что обычно меняют первым

Чаще всего меняют:

* `max_seq_length`
* `per_device_train_batch_size`
* `gradient_accumulation_steps`
* `num_train_epochs`
* `learning_rate`

### Пример более “тяжёлого” обучения

```json
"training": {
  "method": "qlora",
  "max_seq_length": 2048,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 8,
  "num_train_epochs": 3,
  "learning_rate": 0.00005,
  "warmup_ratio": 0.05,
  "logging_steps": 5,
  "save_steps": 100,
  "eval_steps": 100,
  "bf16": true,
  "packing": true,
  "save_total_limit": 2,
  "optim": "adamw_8bit"
}
```

---

## `lora`

Пример:

```json
"lora": {
  "r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.0,
  "bias": "none",
  "use_gradient_checkpointing": "unsloth",
  "random_state": 3407,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
}
```

### Что означает

* `r` — ранг LoRA
* `lora_alpha` — scale для LoRA
* `lora_dropout` — dropout
* `bias` — как обрабатывать bias
* `use_gradient_checkpointing` — экономия памяти
* `random_state` — seed
* `target_modules` — какие слои адаптировать

### Когда менять

Менять нужно в основном при переходе на другую архитектуру модели, потому что у разных моделей список target modules отличается.

### Пример для более широкого покрытия

```json
"lora": {
  "r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "bias": "none",
  "use_gradient_checkpointing": "unsloth",
  "random_state": 3407,
  "target_modules": [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
  ]
}
```

---

## `outputs`

Пример:

```json
"outputs": {
  "base_dir": "/output"
}
```

### Что означает

Это корневая папка, куда будут складываться результаты job.

Дальше сервис сам создаёт подкаталоги:

* `logs`
* `lora`
* `checkpoints`
* `metrics`
* `merged`
* `quantized`
* `evaluation`
* `downloads`

### Важно

`/output` — это путь внутри контейнера.
В локальном запуске он проброшен из `local-run/output`.

---

## `postprocess`

Пример:

```json
"postprocess": {
  "merge_lora": true,
  "save_merged_16bit": true,
  "run_awq_quantization": false
}
```

### Что означает

* `merge_lora` — объединять ли LoRA с базовой моделью
* `save_merged_16bit` — сохранять ли merged-модель
* `run_awq_quantization` — делать ли AWQ-квантование

### Когда менять

Если нужна только LoRA-адаптация без merged-модели, можно отключить:

```json
"postprocess": {
  "merge_lora": false,
  "save_merged_16bit": false,
  "run_awq_quantization": false
}
```

Это уменьшит нагрузку на диск и время завершения job.

---

## `evaluation`

В `defaults` задана базовая evaluation-конфигурация, а в конкретном job она переопределяется.

Базовый смысл такой:

```json
"evaluation": {
  "enabled": true,
  "engine": "vllm",
  "target": "merged",
  "batch_size": 1,
  "max_num_seqs": 1,
  "max_num_batched_tokens": 512,
  "max_model_len": 512,
  "tensor_parallel_size": 1,
  "gpu_memory_utilization": 0.64,
  "max_new_tokens": 8,
  "temperature": 0.0,
  "do_sample": false,
  "score_min": 0.0,
  "score_max": 5.0,
  "parsing_regex": "^\\s*([0-5](?:\\.0)?)\\s*$",
  "dataset": {
    "source": "local",
    "path": "/data/eval.jsonl",
    "format": "jsonl",
    "task": "score_prediction",
    "prompt_field": "input",
    "score_field": "output",
    "tags_field": "details.hash"
  }
}
```

### Основные поля

* `enabled` — включена ли evaluation
* `engine` — движок evaluation, здесь `vllm`
* `target` — что оценивать:

  * `merged`
  * `lora`
  * `auto`
* `batch_size` — сколько prompt отправлять за раз
* `max_num_seqs` — лимит параллельных последовательностей в vLLM
* `max_num_batched_tokens` — лимит токенов на батч
* `max_model_len` — максимальная длина контекста в eval
* `tensor_parallel_size` — tensor parallelism
* `gpu_memory_utilization` — сколько GPU-памяти отдавать vLLM
* `max_new_tokens` — максимум токенов ответа
* `temperature` и `do_sample` — режим генерации
* `score_min`, `score_max` — допустимый диапазон оценки
* `parsing_regex` — как извлекать числовой score из ответа модели

### Когда менять

Если evaluation падает по памяти, уменьшай:

* `batch_size`
* `max_num_seqs`
* `max_num_batched_tokens`
* `max_model_len`
* `gpu_memory_utilization`

Если модель должна выдавать только число от 0 до 5, текущий `parsing_regex` подходит хорошо.

---

## `evaluation.dataset`

Пример:

```json
"dataset": {
  "source": "local",
  "path": "/data/eval.jsonl",
  "format": "jsonl",
  "task": "score_prediction",
  "prompt_field": "input",
  "score_field": "output",
  "tags_field": "details.hash"
}
```

### Что означает

* `source` — откуда брать eval dataset
* `path` — путь внутри контейнера
* `format` — `json` или `jsonl`
* `task` — тип evaluation
* `prompt_field` — поле с prompt
* `score_field` — поле с эталонным score
* `tags_field` — путь к тегам

### Поддерживаемые задачи

#### `score_prediction`

Модель получает prompt и должна вернуть score.

Пример строки в `eval.jsonl`:

```json
{"input":"Оцени ответ ученика","output":4,"details":{"hash":["math","easy"]}}
```

Здесь:

* `input` → prompt
* `output` → reference score
* `details.hash` → теги

#### `judge`

В этом режиме нужны вопрос и ответ кандидата, а модель работает как judge. Тогда конфиг будет другой:

```json
"dataset": {
  "source": "local",
  "path": "/data/eval.jsonl",
  "format": "jsonl",
  "task": "judge",
  "question_field": "question",
  "answer_field": "candidate_answer",
  "score_field": "reference_score",
  "max_score_field": "max_score",
  "tags_field": "hash_tags"
}
```

Пример строки:

```json
{
  "question": "Сколько будет 2+2?",
  "candidate_answer": "4",
  "reference_score": 5,
  "max_score": 5,
  "hash_tags": ["math"]
}
```

---

## `upload`

Пример:

```json
"upload": {
  "enabled": false
}
```

### Что означает

Блок отвечает за загрузку артефактов наружу.

Для локального режима у тебя это выключено, и это нормально.

Если включать upload, можно отправлять:

* логи
* итоговый config
* train metrics
* evaluation summary
* архивы lora/merged/full output

---

## `huggingface`

Пример:

```json
"huggingface": {
  "enabled": false
}
```

### Что означает

Публикация моделей и метаданных в Hugging Face.

Для локального режима обычно выключено.

Если включать, понадобятся:

* `HF_TOKEN`
* repo id
* параметры publish/upload

---

## `reporting`

Пример:

```json
"reporting": {
  "status": { "enabled": false },
  "progress": { "enabled": false },
  "final": { "enabled": false },
  "logs": { "enabled": false }
}
```

### Что означает

Это отправка callback-ов наружу:

* status
* progress
* final
* logs

Для `local-run` всё выключено, что логично.

---

# Разбор блока `jobs`

В `jobs` лежит список конкретных запусков.
У тебя сейчас один job:

```json
{
  "job_id": "local_run",
  "job_name": "local-run",
  "dataset": {
    "source": "local",
    "train_path": "/data/train.jsonl",
    "format": "instruction_output",
    "input_field": "input",
    "output_field": "output"
  },
  "evaluation": {
    "...": "переопределение defaults"
  }
}
```

---

## `job_id`

```json
"job_id": "local_run"
```

Технический идентификатор job.

---

## `job_name`

```json
"job_name": "local-run"
```

Человекочитаемое имя запуска. Оно попадёт в имена папок и артефактов.

---

## `dataset`

Пример:

```json
"dataset": {
  "source": "local",
  "train_path": "/data/train.jsonl",
  "format": "instruction_output",
  "input_field": "input",
  "output_field": "output"
}
```

### Что означает

* `source` — откуда брать train dataset
* `train_path` — путь к train файлу
* `format` — формат train dataset
* `input_field` — где лежит prompt/input
* `output_field` — где лежит ответ/target

### Поддерживаемые форматы

#### `instruction_output`

Ожидаются поля input/output.

Пример строки:

```json
{"input":"Объясни закон Ома","output":"Закон Ома описывает связь напряжения, тока и сопротивления."}
```

#### `prompt_completion`

Пример:

```json
{"input":"Вопрос: столица Франции?\nОтвет: ","output":"Париж"}
```

#### `messages`

Пример:

```json
{
  "messages": [
    {"role":"system","content":"Ты полезный ассистент"},
    {"role":"user","content":"Привет"},
    {"role":"assistant","content":"Привет! Чем помочь?"}
  ]
}
```

Тогда конфиг будет таким:

```json
"dataset": {
  "source": "local",
  "train_path": "/data/train.jsonl",
  "format": "messages",
  "messages_field": "messages"
}
```

---

## Почему `evaluation` повторяется внутри job

Потому что `jobs[0].evaluation` переопределяет значения из `defaults.evaluation`.

Это удобно: можно держать безопасные базовые параметры в `defaults`, а для конкретного запуска поднять производительность.

Например, в `defaults` evaluation настроен консервативно:

```json
"batch_size": 1,
"max_num_seqs": 1,
"max_num_batched_tokens": 512,
"gpu_memory_utilization": 0.64
```

А в самом job включён более быстрый вариант:

```json
"batch_size": 32,
"max_num_seqs": 8,
"max_num_batched_tokens": 8192,
"gpu_memory_utilization": 0.9
```

---

# Что чаще всего меняют в `job.local.json`

Обычно меняют вот эти блоки:

## 1. Путь к модели

```json
"model": {
  "local_path": "/app"
}
```

Если собрали trainer на другом базовом образе и модель лежит не в `/app`, путь надо поменять.

---

## 2. Путь к train и eval датасетам

```json
"train_path": "/data/train.jsonl"
"path": "/data/eval.jsonl"
```

Если у тебя другие имена файлов, меняешь их здесь.

---

## 3. Длину контекста

```json
"max_seq_length": 256
```

Это и в `model`, и в `training` стоит согласовать с возможностями модели и GPU.

---

## 4. Параметры обучения

Чаще всего:

```json
"per_device_train_batch_size"
"gradient_accumulation_steps"
"num_train_epochs"
"learning_rate"
```

---

## 5. Параметры evaluation

Если мало памяти — уменьшаешь:

```json
"batch_size"
"max_num_seqs"
"max_num_batched_tokens"
"max_model_len"
"gpu_memory_utilization"
```

---

# Примеры готовых вариантов

## Минимальный локальный job для `instruction_output`

```json
{
  "defaults": {
    "mode": "local",
    "model": {
      "source": "local",
      "local_path": "/app",
      "load_in_4bit": true,
      "dtype": "bfloat16",
      "max_seq_length": 512
    },
    "training": {
      "method": "qlora",
      "max_seq_length": 512,
      "per_device_train_batch_size": 1,
      "gradient_accumulation_steps": 4,
      "num_train_epochs": 1,
      "learning_rate": 0.0001,
      "bf16": true,
      "packing": false,
      "logging_steps": 1,
      "save_steps": 50,
      "eval_steps": 50,
      "save_total_limit": 1,
      "optim": "adamw_8bit"
    },
    "lora": {
      "r": 8,
      "lora_alpha": 16,
      "lora_dropout": 0.0,
      "bias": "none",
      "use_gradient_checkpointing": "unsloth",
      "random_state": 3407,
      "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    },
    "outputs": {
      "base_dir": "/output"
    },
    "postprocess": {
      "merge_lora": true,
      "save_merged_16bit": true,
      "run_awq_quantization": false
    },
    "evaluation": {
      "enabled": false
    },
    "upload": {
      "enabled": false
    },
    "huggingface": {
      "enabled": false
    },
    "reporting": {
      "status": { "enabled": false },
      "progress": { "enabled": false },
      "final": { "enabled": false },
      "logs": { "enabled": false }
    }
  },
  "jobs": [
    {
      "job_id": "demo_job",
      "job_name": "demo-job",
      "dataset": {
        "source": "local",
        "train_path": "/data/train.jsonl",
        "format": "instruction_output",
        "input_field": "input",
        "output_field": "output"
      }
    }
  ]
}
```

---

## Локальный job с `messages`

```json
{
  "jobs": [
    {
      "job_id": "chat_job",
      "job_name": "chat-job",
      "dataset": {
        "source": "local",
        "train_path": "/data/train_messages.jsonl",
        "format": "messages",
        "messages_field": "messages"
      }
    }
  ]
}
```

---

## Локальный job с более осторожной evaluation

```json
"evaluation": {
  "enabled": true,
  "engine": "vllm",
  "target": "merged",
  "batch_size": 1,
  "max_num_seqs": 1,
  "max_num_batched_tokens": 512,
  "max_model_len": 512,
  "tensor_parallel_size": 1,
  "gpu_memory_utilization": 0.6,
  "max_new_tokens": 8,
  "temperature": 0.0,
  "do_sample": false,
  "score_min": 0.0,
  "score_max": 5.0,
  "parsing_regex": "^\\s*([0-5](?:\\.0)?)\\s*$",
  "dataset": {
    "source": "local",
    "path": "/data/eval.jsonl",
    "format": "jsonl",
    "task": "score_prediction",
    "prompt_field": "input",
    "score_field": "output",
    "tags_field": "details.hash"
  }
}
```

---

# Практическая рекомендация

Для повседневной работы обычно меняют только:

* `model.local_path`
* `dataset.train_path`
* `evaluation.dataset.path`
* `training.max_seq_length`
* `training.per_device_train_batch_size`
* `training.gradient_accumulation_steps`
* `training.num_train_epochs`
* `evaluation.batch_size`
* `evaluation.max_num_seqs`
* `evaluation.max_num_batched_tokens`