# Quick Start — первый запуск Family Finance Assistant

Пошаговая инструкция «с нуля до работающего бота». **Время:** 30-45 минут (большая часть — ожидание загрузки Docker-образов).

## 📋 Что у тебя должно быть до начала

### Софт на машине

```bash
# Проверить версии:
python3 --version    # 3.12+
docker --version     # 24+
docker compose version  # v2+ (не docker-compose)
git --version
curl --version
just --version
```

### Что нужно получить онлайн

- **OpenRouter API key**
- **Telegram bot token** —  @BotFather
- **Твой Telegram user ID** — узнать у @userinfobot

Эти шаги ниже.

### Ресурсы машины

- **RAM**: минимум 6 GB свободной (ClickHouse + Postgres + другие контейнеры)
- **Диск**: 4-5 GB на образы Docker
- **Порты должны быть свободны**: `5432`, `3001`, `6379`, `9090`, `9091`. Проверить:
  ```bash
  ss -tlnp | grep -E ':(5432|3001|6379|9090|9091)\s'
  # Должно быть пусто
  ```

---

## Шаг 1 — Получить проект

```bash
# Куда положишь проект — на твой выбор. Пример:
mkdir -p ~/_projects && cd ~/_projects

git clone git@github.com:Carobey/bot-ff.git
cd bot-ff

Если проект передан архивом, распакуй его в отдельную директорию и дальше выполняй команды из
корня проекта.
```

---

## Шаг 2 — Поставить Python-зависимости

```bash
just install
```

Это запустит `uv sync` — создаст `.venv/` и поставит все зависимости из `pyproject.toml`.

---

## Шаг 3 — Получить OpenRouter API key

3.1. Зайти на **https://openrouter.ai** → **Sign in** (можно через Google).

3.2. Создать API key:
- Меню → **Keys** → **Create Key**
- Скопируй ключ **сразу** — он начинается с `sk-or-v1-...`, показывается один раз.

---

## Шаг 4 — Создать Telegram-бота

> Создавай **отдельного бота**, не используй существующих.

### 4.1. Бот у @BotFather

1. Открой Telegram → найди **@BotFather** → `/start`

2. `/newbot` → BotFather задаст 2 вопроса:

   **Display name** (что видно в чатах):
   ```
   Family Finance
   ```

   **Username** (должен заканчиваться на `bot`, уникальный во всём Telegram):
   ```
   например ff_username_bot
   ```

3. BotFather пришлёт сообщение типа:
   ```
   Done! Congratulations on your new bot...
   Use this token to access the HTTP API:

   7234567890:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw
   ```

   **Этот токен и нужен.** Скопируй целиком в надёжное место.

### 4.2. Узнать свой Telegram user ID

Это число — твой уникальный ID, не username. Нужен для whitelist'а (чтобы посторонние не могли писать боту).

1. В Telegram найди **@userinfobot** (любой, обычно популярный с синей галочкой)
2. Нажми `/start`
3. Бот пришлёт что-то типа:
   ```
   👤 User
   Id: 123456789
   First Name: Yuri
   Username: @yourname
   Language: ru
   ```

Пользователей можно указать через запятую.

---

## Шаг 5 — Заполнить `.env`

```bash
just init-env
# Это создаст .env из .env.example если его нет.

# Открыть в редакторе:
nano .env
```

**Минимум что нужно отредактировать (3 строки):**

```bash
# от BotFather:
TELEGRAM_BOT_TOKEN=7234567890:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw

# твой ID от @userinfobot, без пробелов вокруг запятых:
TELEGRAM_ALLOWED_USER_IDS=123456789

# от OpenRouter:
OPENROUTER_API_KEY=sk-or-v1-abc123def456...
```

`TELEGRAM_ALLOWED_USER_IDS` обязателен. Если оставить его пустым, бот никого не пустит.

---

## Шаг 6 — Поднять инфраструктуру

```bash
just up
```

Это запустит `docker compose up -d` для контейнеров:
- `ff-postgres` - основная БД проекта
- `ff-lf-postgres` - БД для LangFuse
- `ff-lf-clickhouse` - аналитическая БД для traces
- `ff-lf-redis` - очередь LangFuse
- `ff-lf-minio` - S3-совместимое хранилище для LangFuse
- `ff-lf-worker` - фоновый обработчик LangFuse
- `ff-lf-web` - UI LangFuse

**Первый запуск:** - Docker качает образы (~3 GB).

Проверить статус:
```bash
docker compose ps
```

Все сервисы должны быть `Up`, у Postgres'ов — `(healthy)`. Если какой-то в `Restarting` — смотри логи:
```bash
docker compose logs <service-name> --tail 50
# Например:
docker compose logs langfuse-clickhouse --tail 50
```

### Типичные проблемы

| Симптом | Причина | Решение |
|---|---|---|
| `langfuse-clickhouse` рестартится | Не хватает памяти | Добавь swap или закрой что-то жрущее. ClickHouse требует ~2 GB |
| `langfuse-web` Unhealthy первые 1-2 мин | Ждёт миграций worker'а | Подожди 60-120 сек, проверь снова |
| Порт 5432 занят | Локальный Postgres запущен | `sudo systemctl stop postgresql` или меняй порт в `docker-compose.yml` |
| `permission denied: docker.sock` | `usermod -aG docker` не применился | Logout+login (важно). Проверь: `groups` должно содержать `docker` |
| MinIO не стартует | Старый volume с другими ключами | `just nuke` (!!! удалит все контейнеры от ff) и `just up` заново |

---

## Шаг 7 — Проверить инфру

```bash
just check-langfuse
```
Ожидание: `✅ LangFuse healthy`. Если ❌ — `docker compose logs langfuse-web --tail 100`.

```bash
just check-llm
```
Ожидание:
```
✅ OpenRouter key ok
   label:        family-finance-diploma
   usage:        $0.0000
   limit:        5.0
   is_free_tier: False
   supervisor openai/gpt-5.4: ✅
   worker     google/gemini-2.5-flash: ✅
```

Если `is_free_tier: True` — кредиты не дошли, проверь страницу Credits на OpenRouter.

Если модели `❌ не найдена` — OpenRouter мог переименовать (см. `just list-cheap-models`). Обнови `LLM_SUPERVISOR_MODEL` / `LLM_WORKER_MODEL` в `.env`.

---

## Шаг 8 — Войти в LangFuse UI

```bash
http://localhost:3001
```

Логин:
- **Email**: `admin@local.dev`
- **Password**: `admin12345`

(Эти значения из `.env`, переменные `LANGFUSE_INIT_USER_EMAIL` и `LANGFUSE_INIT_USER_PASSWORD`.)

После входа должен увидеть organization **FamilyFinance** и project **family-finance** уже созданными (init-параметры в docker-compose).

---

## Шаг 9 — Запустить бота

В **отдельном терминале** (текущий понадобится для логов):

```bash
just run
```

Должен увидеть:
```
2026-05-27T19:42:01 [info] startup.begin     service=family-finance env=dev
2026-05-27T19:42:01 [info] ✅ LangFuse client initialized: http://localhost:3001
2026-05-27T19:42:02 [info] Connecting checkpointer: localhost:5432/family_finance
2026-05-27T19:42:02 [info] ✅ PostgresSaver ready
2026-05-27T19:42:02 [info] startup.ready     bot=ff_yuri_diploma_bot
```

**Если падает:**
- `aiogram.exceptions.TelegramUnauthorizedError` → токен неправильный
- `OperationalError: connection refused` → Postgres не успел, подожди 5 сек и `just run` снова
- `pydantic.ValidationError: telegram_bot_token` → проверь что в `.env` нет лишних пробелов / кавычек

Бот работает в foreground — оставь терминал открытым.

---

## Шаг 10 — Первое сообщение боту 🎉

10.1. В Telegram найди своего бота по username (`@ff_yuri_bot` или как назвал).

10.2. Нажми **Start** (или напиши `/start`).

10.3. Должен прийти ответ:
```
👋 Привет! Я — финансовый помощник семьи.

Сейчас Phase 0 — инфра-чек.
Просто напиши мне что-нибудь, и проверь что в LangFuse появился trace:
http://localhost:3001/project/ff-project/traces

Phase 1 — загрузка CSV выписки и распознавание чеков — скоро.
```

10.4. Напиши **любое сообщение**, например:
```
Привет, как дела?
```

10.5. Через 1-3 секунды бот должен ответить через GPT-5.4 (что-то дружелюбное и краткое).

10.6. В логе бота (терминал с `just run`) увидишь активность.

---

## Шаг 11 — Проверить trace в LangFuse 🎯

Это ключевая проверка — **именно эти скриншоты нужны Эмилю** на ревью 4 июня.

11.1. Открой `http://localhost:3001/project/ff-project/traces`

11.2. Должна появиться новая trace (refresh страницу если не видна). Кликни на неё.

11.3. Что должно быть в trace:
- **User ID**: твой Telegram ID (как число)
- **Session ID**: `tg:<chat_id>`
- **Tags**: `telegram`, `phase0`
- **Spans** (раскрывающиеся узлы):
  - LangGraph: `supervisor` node
  - LLM: вызов `openai/gpt-5.4` с input/output/tokens
- **Cost**: реальная стоимость в долларах (≈ $0.0005-0.002 за одно сообщение)
- **Metadata**: `telegram_user_id`, `telegram_chat_id`

11.4. Сделай скриншот. Положи в `docs/screenshots/phase0-first-trace.png`. Это первый артефакт для презентации.

---

## ✅ Phase 0 Definition of Done

Если всё работает:
- [x] Бот отвечает на `/start` и любые сообщения
- [x] В LangFuse появляются traces с моделью, стоимостью, тегами
- [x] `just lint && just test` — зелёное
- [x] Все docker-сервисы healthy

**Сделай git init и первый коммит:**
```bash
git init
git add .
git commit -m "Phase 0: foundation"

# Создать приватный репо на GitHub: github.com/new
# Имя: family-finance
# Visibility: Private (обязательно!)
git remote add origin git@github.com:yourname/family-finance.git
git push -u origin main

# Включить pre-commit hooks:
uv run pre-commit install
```

Поздравляю, **Phase 0 done**! 🎓

---

## 🛟 Шпаргалка ежедневных команд

```bash
# Полный старт после перезагрузки машины:
just up && just check-langfuse && just check-llm && just run

# Остановить всё:
just down

# Перезапуск конкретного сервиса:
docker compose restart langfuse-web

# Логи одного сервиса (последние 50 строк):
just logs langfuse-web

# Логи бота (он же в foreground, но если ушёл на background):
# Просто перезапусти его: Ctrl+C → just run

# Линт + тесты:
just lint && just test

# Что я наизменял:
git status
git diff

# Полный smoke check:
just smoke

# Удалить ВСЁ (БД, volumes) — для чистого старта:
just nuke   # ⚠️ удалит все данные!
```

---

## 🐛 Что делать если совсем сломалось

1. **Базовый перезапуск:**
   ```bash
   just down
   docker compose ps  # должно быть пусто
   just up
   ```

2. **Сброс volumes без удаления образов** (теряется только локальная БД):
   ```bash
   just nuke
   just up
   ```

3. **Проверить что .env корректен:**
   ```bash
   # Не должно быть кавычек/пробелов/комментариев на той же строке
   grep -E '^(TELEGRAM|OPENROUTER)' .env
   ```

4. **Проверить ресурсы:**
   ```bash
   docker stats --no-stream    # каждый контейнер RAM
   df -h /var/lib/docker        # место на диске
   free -h                      # свободная RAM
   ```

5. **Если ничего не помогло — спросить Claude** (Code или этот чат), приложив:
   - `docker compose ps`
   - `docker compose logs <падающий-сервис> --tail 100`
   - содержимое `.env` **с замаскированными токенами** (`sk-or-v1-***`)

---

См. `docs/BACKLOG.md` раздел Phase 1.
