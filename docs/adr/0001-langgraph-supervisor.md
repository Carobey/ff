# 0001 - LangGraph для оркестрации supervisor-графа

**Status:** Accepted

**Date:** 2026-05-26

**Updated:** 2026-06-15

## Контекст

Ассистент объединяет импорт выписок, категоризацию, уточнение неоднозначных операций,
обработку чеков, запросы к ledger, поведенческие паттерны, подписки, бюджеты,
финансовые рекомендации и оценку социальных налоговых вычетов. Сценариям нужны:

- общий типизированный state и история сообщений;
- явный и тестируемый routing для известных интентов;
- fallback-планирование свободных и составных финансовых запросов;
- параллельное выполнение независимых read-only секций;
- human-in-the-loop паузы перед записью и для недостающих входных данных;
- persistence между сообщениями и после рестарта;
- ограничение роста длинной истории диалога;
- возможность добавлять специализированные узлы без зависимости domain-слоя от LLM;
- единая точка подключения observability.

Самописный цикл оркестрации быстро дублировал бы state management, checkpointing и
resume semantics. Полностью LLM-управляемый supervisor был бы дороже и менее
предсказуем для интентов и числовых расчетов, которые можно определить кодом.

## Решение

Использовать **LangGraph 1.2+** и строить основной workflow через `StateGraph`.

Текущий контракт:

- `FinanceState` - `TypedDict` с reducer-ами для сообщений, транзакций и активного
  набора clarification-вопросов, а также accumulator-ом multi-intent секций;
- `compact` перед supervisor сворачивает старую часть длинного треда в rolling summary;
- trusted pending upload state маршрутизируется первым, затем `supervisor_node`
  применяет injection guard к последнему пользовательскому сообщению;
- clarification и известные одиночные интенты выбираются Python-правилами;
- явный составной запрос строит ordered plan и запускает read-only секции через
  `Send`; `section_worker` fan-out сходится в `synthesizer`;
- если keyword fast-path не сработал, structured-output LLM-планировщик выбирает
  ноль, одну или несколько финансовых секций; ноль означает small talk;
- LLM-планировщик выбирает только маршрут: периоды, SQL-агрегации, налоговые суммы
  и итоговые секции считаются кодом;
- импорт имеет второй branch point: `ingest -> categorizer | END`;
- `ingest` вызывает `interrupt()` после чистого парсинга и до `add_many`; resume
  получает подтверждение пользователя;
- `tax` может вызвать `interrupt()` для дохода и признаков дорогостоящего лечения
  или обучения детей;
- specialist-нода формирует единственный пользовательский ответ и завершает run;
- `AsyncPostgresSaver` хранит state по `thread_id = "tg:<chat_id>"`;
- граф компилируется без checkpointer только для offline-инструментов вроде
  `just printgraph`; runtime бота всегда использует PostgresSaver.

Скомпилированный граф содержит `compact`, `supervisor`, `ingest`, `categorizer`,
`clarify`, `ledger`, `receipt`, `coach`, `subscriptions`, `budgets`, `advisor`,
`tax`, `section_worker` и `synthesizer`. Недельный `digest` запускается командой
или APScheduler вне LangGraph.

## Обоснование

- Явные edges и route-функции легко покрываются unit-тестами и eval-кейсами.
- Typed state делает межузловые контракты видимыми и сериализуемыми.
- Postgres checkpointer сохраняет диалог, interrupts и pending clarification между invoke.
- Детерминированный routing не тратит LLM-вызов на известные интенты.
- Ограниченный LLM planner улучшает recall без передачи ему вычислений и write access.
- `Send` позволяет параллельно собрать независимые секции, а reducer и явный `order`
  делают fan-in предсказуемым.
- `interrupt/resume` отделяет preview и вопросы от необратимой записи или расчета.
- Compaction ограничивает размер checkpoint и trace, сохраняя последние сообщения.
- LangGraph callbacks позволяют связать один invoke с trace в LangFuse.
- Domain и application ports не импортируют LangGraph.

## Рассмотренные альтернативы

### Самописный orchestrator

Проще на старте, но потребовал бы собственной реализации persistence, reducer-ов,
resume semantics и трассировки переходов.

### Полностью LLM-управляемый supervisor / ReAct

Гибче для открытого набора инструментов, но добавляет стоимость и
недетерминированность. Выбран гибрид: правила остаются fast-path, а LLM используется
как ограниченный planner только после промаха правил. ReAct можно добавлять локально
внутри отдельного specialist, не меняя topology всего графа.

### Последовательный запуск всех specialist-нод

Проще оркестрировать, но увеличивает latency составного ответа и не использует
независимость read-only запросов. Выбран fan-out через `Send` с детерминированным
fan-in.

### CrewAI / AutoGen

Ориентированы на диалог автономных агентов. Здесь важнее контролируемый workflow,
общий state и предсказуемые переходы.

## Следствия

Положительные:

- topology видна в коде и генерируется командой `just printgraph`;
- state переживает рестарты процесса;
- import/tax workflows можно безопасно приостанавливать и продолжать;
- составные запросы не теряют вторичные интенты;
- новые сценарии добавляются как node + route + тест;
- LLM не является обязательным маршрутизатором и не считает финансовые числа.

Отрицательные и ограничения:

- schema `FinanceState` и persisted checkpoints требуют совместимых изменений;
- один `thread_id` на Telegram-чат предполагает private-chat модель;
- Postgres становится обязательной runtime-зависимостью;
- детерминированные intent-функции нужно поддерживать при расширении формулировок;
- LLM planner добавляет latency и стоимость для keyword miss;
- parallel workers требуют concurrency-safe lifecycle общих клиентов, включая MCP;
- interrupt payload и resume handler образуют versioned контракт между graph и bot;
- compaction является lossy и отправляет старую историю в LLM для суммаризации;
- LangGraph не является security boundary: авторизация и tenant isolation реализуются
  интерфейсом и repository/MCP-слоем.

## Правила развития

- Не помещать бизнес-инварианты в route-функции: они должны оставаться в domain или
  application/infrastructure сервисах.
- Предпочитать детерминированный route, когда intent однозначен.
- LLM planner не должен получать write tools или считать итоговые суммы.
- В multi-intent fan-out включать только независимые read-only секции; порядок ответа
  задавать полем `order`, а не порядком завершения задач.
- Код до `interrupt()` должен быть идемпотентным и не иметь необратимых side effects.
- Для каждого interrupt определить payload schema, bot renderer и resume-тест.
- Не добавлять промежуточные `AIMessage`, если specialist сам формирует ответ.
- Для нового state-поля определить merge semantics и проверить совместимость
  существующих checkpoints.
- Любой новый branch покрывать unit-тестом routing; planner и пользовательские
  интенты также покрывать eval-кейсами.
