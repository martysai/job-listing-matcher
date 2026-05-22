# Evaluation Pipeline — Инструкция по запуску

## Файлы проекта

Для работы нужны четыре файла, расположенные в одной папке:

| Файл | Назначение |
|---|---|
| `evaluation_pipeline.py` | Основной код: LLM-вызовы, проверки, отчёты |
| `dataset.py` | Тест-кейсы |
| `prompt_v1.py` | Промпт версии 1 (исходный) |
| `prompt_v2.py` | Промпт версии 2 (улучшенный) |

---

## Часть I — Запуск в Google Colab

### Шаг 1. Добавить API-ключ в хранилище секретов

> **Почему не в коде.** Ключ, вписанный напрямую в ячейку или файл, сохраняется в истории ноутбука и может оказаться в открытом доступе при публикации. Colab Secrets хранит ключи изолированно — они не отображаются в выводе и не попадают в историю.

1. В левой панели Colab нажать иконку **🔑 Secrets**
2. Нажать **+ Add new secret**
3. Заполнить поля:

| Поле | Значение |
|---|---|
| **Name** | Имя переменной вашего провайдера (см. таблицу ниже) |
| **Value** | Ваш API-ключ |

4. Переключить тумблер **Notebook access** → **ON**

**Имена переменных по провайдерам:**

| Провайдер | Name |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` |

---

### Шаг 2. Загрузить файлы проекта

1. В левой панели нажать иконку **📁 Files**
2. Нажать кнопку со стрелкой вверх — **Upload to session storage**
3. Выбрать все четыре файла одновременно
4. Убедиться, что они появились в корне `/content/`

> **Важно.** Файлы хранятся только в рамках текущей сессии. При повторном открытии ноутбука загрузку нужно повторить.

---

### Шаг 3. Выбрать провайдера и модель

Открыть `evaluation_pipeline.py` и в начале файла найти блок `▶ CONFIGURATION`. Изменить две строки:

```python
MODEL: str = "gpt-4o-mini"           # ← модель
API_KEY_ENV: str = "OPENAI_API_KEY"  # ← имя переменной из Secrets
```

**Примеры для других провайдеров:**

```python
# Mistral
MODEL       = "mistral/mistral-large-latest"
API_KEY_ENV = "MISTRAL_API_KEY"

# Anthropic
MODEL       = "claude-3-5-haiku-20241022"
API_KEY_ENV = "ANTHROPIC_API_KEY"

# Google Gemini
MODEL       = "gemini/gemini-1.5-flash"
API_KEY_ENV = "GEMINI_API_KEY"
```

---

### Шаг 4. Создать ноутбук и выполнить ячейки по порядку

**Ячейка 1 — установка зависимостей** *(один раз за сессию)*

```python
!pip install -q litellm pandas tabulate
```

**Ячейка 2 — загрузка ключа из Secrets в окружение**

```python
from google.colab import userdata
import os

# Имя должно совпадать с тем, что указано в Secrets на шаге 1
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
```

> Colab Secrets не загружают ключ в `os.environ` автоматически. Эта ячейка явно переносит значение туда, откуда его читает `litellm`. Сам ключ не отображается в выводе.

**Ячейка 3 — запуск pipeline**

```python
%run evaluation_pipeline.py
```

---

### Шаг 5. Читать результаты

После запуска ячейки 3 вывод появляется последовательно:

```
Running evaluation pipeline  (model: gpt-4o-mini)

[1/22] TC-01 · prompt v1 … ✓
[2/22] TC-01 · prompt v2 … ✓
...
[22/22] TC-11 · prompt v2 … ✓
```

Затем автоматически печатаются три раздела:

| Раздел | Что показывает |
|---|---|
| **CHECKS REGISTRY** | Таблица всех 24 проверок с типами |
| **AGGREGATE SCORES** | Общий счёт v1 vs v2 с прогресс-барами |
| **V1 vs V2 — PER TEST CASE** | Результат по каждому кейсу и дельта |
| **DETAIL: TC-01 … TC-11** | Детализация каждого кейса: какие проверки прошли / упали |

В конце вывода отображаются цветные таблицы DataFrame (зелёный = 100%, жёлтый = ≥ 70%, красный = < 70%).

---

### Шаг 6. Дополнительные команды в ноутбуке

После выполнения `%run` все функции и переменные доступны в следующих ячейках:

```python
# Детализация одного кейса
print_detail(results, "TC-06")

# Перепечатать полный отчёт
print_full_report(results)

# Сравнительная таблица как DataFrame
comparison_df

# Сырые данные первого запуска
results[0]
```

---

### Шаг 7. Управление тест-кейсами

Открыть `dataset.py` и отредактировать напрямую.

**Деактивировать кейс** — не удалять, а пропустить:

```python
{
    "id": "TC-08",
    "active": False,  # ← было True
    ...
}
```

**Добавить новый кейс** — скопировать любой существующий и заменить поля:

```python
{
    "id": "TC-12",
    "active": True,
    "target_field": "candidate_description.skills",
    "rule_type": "positive_extraction",
    "input": "I have experience with FastAPI and PostgreSQL.",
    "assertions": {
        "candidate_description.skills": ["FastAPI", "PostgreSQL"],
    },
    "notes": "Проверка извлечения навыков из короткого описания.",
},
```

После изменений — перезагрузить файл и перезапустить ячейку с `%run`.

> В Colab повторно загрузить изменённый `dataset.py` через **📁 Files → Upload** и снова выполнить ячейку 3.

---

### Шаг 8. Добавить новую версию промпта

1. Создать файл `prompt_v3.py` по образцу `prompt_v1.py`
2. Вставить полный текст промпта, убедиться что он заканчивается строкой:

```
job_request = <<<JOB_REQUEST>>>
```

3. Загрузить файл в Colab
4. В `evaluation_pipeline.py` добавить две строки в блок **External modules**:

```python
from prompt_v3 import PROMPT_V3
PROMPT_REGISTRY["v3"] = PROMPT_V3
```

5. Запустить с тремя версиями в новой ячейке:

```python
results = run_pipeline(DATASET, versions=["v1", "v2", "v3"])
print_full_report(results)
```

---
---

## Часть II — Запуск локально (IDE / Jupyter)

### Шаг 1. Подготовить окружение

```bash
# Создать виртуальное окружение (рекомендуется)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# Установить зависимости
pip install litellm pandas tabulate python-dotenv
```

---

### Шаг 2. Создать файл `.env` с API-ключом

Создать файл `.env` в корне папки проекта (рядом с четырьмя `.py` файлами):

```
OPENAI_API_KEY=sk-ваш-ключ-здесь
```

Для других провайдеров — добавить нужную строку:

```
MISTRAL_API_KEY=ваш-ключ
ANTHROPIC_API_KEY=ваш-ключ
GEMINI_API_KEY=ваш-ключ
```

**Защитить ключ от попадания в репозиторий** — добавить `.env` в `.gitignore`:

```
# .gitignore
.env
```

> Файл `.env` читается локально и не передаётся никуда. Никогда не коммитьте его в git и не передавайте другим людям.

---

### Шаг 3. Выбрать провайдера и модель

Открыть `evaluation_pipeline.py`, найти блок `▶ CONFIGURATION` и изменить две строки — так же, как описано в [Шаге 3 для Colab](#шаг-3-выбрать-провайдера-и-модель).

---

### Шаг 4. Запустить pipeline

**Вариант А — как скрипт:**

```bash
python evaluation_pipeline.py
```

Перед запуском добавить загрузку `.env` в начало `evaluation_pipeline.py` (сразу после блока imports):

```python
from dotenv import load_dotenv
load_dotenv()
```

**Вариант Б — в Jupyter Notebook:**

```python
# Ячейка 1 — загрузка ключей из .env
from dotenv import load_dotenv
load_dotenv()

# Ячейка 2 — запуск
%run evaluation_pipeline.py
```

**Вариант В — в VS Code с файлом `.env`:**

VS Code автоматически подхватывает `.env` при запуске Python через кнопку ▶, если в `settings.json` или `.vscode/launch.json` указано:

```json
{
    "python.envFile": "${workspaceFolder}/.env"
}
```

---

### Шаг 5. Читать результаты

Вывод в терминале или ячейке Jupyter идентичен выводу в Colab (см. [Шаг 5 для Colab](#шаг-5-читать-результаты)).

В Jupyter дополнительно отображаются цветные таблицы DataFrame. В терминале — только текстовые таблицы с ANSI-цветами.

---

### Шаги 6–8

Управление тест-кейсами и добавление новых версий промпта — идентично [Шагам 6–8 для Colab](#шаг-6-дополнительные-команды-в-ноутбуке). Разница только в том, что файлы редактируются локально в IDE, перезагружать их вручную не нужно.

---
---

## Справочник: структура проверок

### Типы field-проверок (CHKF)

| Тип | Условие прохождения | Используется для |
|---|---|---|
| `exact` | Значения равны (без учёта регистра у строк) | Скалярные поля: `years_experience`, `education_level`, `salary_min` и др. |
| `set_f1` | F1-мера по множествам = 1.0 | Списки: `skills`, `languages`, `desired_positions` и др. |
| `set_f1_soft` | F1-мера по множествам ≥ 0.5 | `preferred_activities` — допуск на нормализацию глаголов к герундию |
| `set_exact` | Множества идентичны | `employment_type`, `preferred_remote_policy`, `acceptable_remote_policy` |
| `companies_f1` | F1 по парам (industry, location) = 1.0 | `preferred_companies` |

### Инварианты (CHKI)

Проверяются автоматически для каждого ответа модели, независимо от тест-кейса:

| ID | Что проверяет |
|---|---|
| CHKI-01 | `candidate_description` всегда присутствует и является dict |
| CHKI-02 | `job_description` всегда присутствует и является dict |
| CHKI-03 | `preferred_remote_policy ∩ acceptable_remote_policy = ∅` |
| CHKI-04 | Все значения `employment_type` входят в допустимый enum |
| CHKI-05 | Все значения remote policy входят в `{remote, hybrid, onsite}` |
| CHKI-06 | `currency` — трёхбуквенный ISO 4217 код или null |
| CHKI-07 | Если `preferred_work_mode` не null, в нём есть хотя бы один сигнал |

### Что происходит при смене схемы

| Сценарий | Что нужно изменить |
|---|---|
| Добавить поле | `make_empty_output()` + одна запись в `FIELD_CHECKS` |
| Удалить поле | Удалить из `make_empty_output()` + удалить запись из `FIELD_CHECKS` |
| Изменить тип сравнения | Изменить `"type"` в соответствующей записи `FIELD_CHECKS` |
| Добавить тест на новое поле | Добавить в `assertions` нужного тест-кейса в `dataset.py` |
