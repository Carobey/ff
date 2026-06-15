# 0002 - Self-hosted LangFuse для observability и evals

**Status:** Accepted

**Date:** 2026-05-26

**Updated:** 2026-06-15

## Контекст

Для multi-step LangGraph workflow нужно видеть route, LLM-вызовы, tool calls,
interrupt/resume, latency, ошибки и стоимость в разрезе пользователя и диалога.
Отдельно нужен воспроизводимый eval pipeline с датасетами и score-ами для
категоризации, парсинга, security, routing и multi-intent planning.

Трассы содержат финансовый контекст и идентификаторы Telegram, поэтому внешний SaaS
увеличивает число получателей чувствительных данных. При этом само приложение уже
использует OpenRouter, и локальная observability не устраняет этот внешний поток.

Рассматривались LangSmith, Laminar и vendor-neutral OpenTelemetry backend.

## Решение

Использовать **LangFuse self-hosted**.

Локальный stack в `docker-compose.yml`:

- `langfuse-web` и `langfuse-worker`;
- отдельный `langfuse-postgres`;
- ClickHouse;
- Redis;
- MinIO.

Интеграция приложения:

- singleton `Langfuse` и отдельный `CallbackHandler` на каждый LangGraph invoke;
- metadata содержит `langfuse_user_id`, `langfuse_session_id`, tags и trace name;
- Telegram session совпадает с LangGraph `thread_id`;
- callback создается также для resume-вызовов, digest и parsing расписания;
- Graphiti использует instrumented `langfuse.openai.AsyncOpenAI` для LLM, embeddings
  и reranker вызовов;
- graph-ноды отправляют production scores через `score_current_trace`; сейчас это
  `injection_blocked` и `categorization_review_rate`;
- shutdown бота вызывает `flush()`.

Eval pipeline разделен на два режима:

- `just eval` - pytest gate по YAML-кейсам, включая LLM-as-judge;
- `just eval-report` - отдельный долгоживущий процесс, который создает LangFuse
  datasets `ff-<agent>`, запускает experiment и записывает программные score-ы.

Разделение нужно из-за несовместимости lifecycle OTLP exporter с event loop-ами
pytest в текущей конфигурации. Production scores и offline dataset scores имеют
разные назначения и имена: первые мониторят реальные runs, вторые сравнивают версии.

## Обоснование

- Нативная интеграция с LangChain/LangGraph сокращает объем instrumentation-кода.
- Self-host сохраняет observability store под контролем владельца проекта.
- Dataset runs дают сравнение версий prompt/model по одному набору кейсов.
- User/session/tags связывают отдельные node и generation events в один диалог.
- Scores позволяют строить dashboard не только по latency/cost, но и по качественным
  сигналам, не разбирая текст trace вручную.
- Один продукт закрывает runtime tracing и отчетность по evals.

## Рассмотренные альтернативы

### LangSmith

Сильная интеграция с LangGraph, но основной вариант размещения добавляет внешний
SaaS в контур финансовых данных.

### Laminar

Удобен для анализа длинных agent runs, но не выбран из-за меньшего совпадения с
требованиями проекта и уже реализованным eval workflow.

### Только OpenTelemetry

Снижает vendor lock-in, но потребовал бы отдельно собирать UI, prompt/generation
семантику, datasets и scoring workflow.

## Следствия

Положительные:

- runtime traces и eval runs доступны в одном локальном UI;
- можно анализировать latency, ошибки и LLM usage по сессиям;
- eval datasets сохраняются между прогонами;
- приложение не зависит от облачного LangFuse.

Отрицательные и ограничения:

- stack добавляет шесть контейнеров и отдельные persistent volumes;
- LangFuse становится хранилищем чувствительных данных и требует контроля доступа,
  retention, backup и удаления;
- graph callbacks могут сохранять исходные input/state до PII-маскирования LLM-вызова;
- interrupt payload, ответы resume и rolling summary могут попасть в trace;
- self-hosting LangFuse не предотвращает отправку prompts, receipt images и Graphiti
  payload в OpenRouter, включая `:online` web-search;
- локальные dev credentials и опубликованные порты неприемлемы для production;
- часть eval score-ов в dataset experiment детерминированная: `llm_judge` выполняется
  только в pytest gate;
- `emit_score` работает fail-soft: потеря score не ломает пользовательский workflow,
  поэтому monitoring должен отдельно обнаруживать пропуски telemetry.

## Эксплуатационные правила

- В production заменить все default secrets и закрыть сервисы внутренней сетью.
- Ограничить доступ к UI и volumes как к финансовой БД.
- Не добавлять secrets, полные банковские файлы или изображения в metadata.
- Не помещать в score comments исходные пользовательские payload.
- Определить retention и процедуру удаления traces до реального использования.
- После изменения prompt/model запускать `just eval`; для отчетного run -
  `just eval-report`.
- При добавлении production score определить тип, диапазон, момент отправки и поведение
  при отсутствии активного trace.
- При смене backend сохранить metadata contract и eval dataset semantics.
