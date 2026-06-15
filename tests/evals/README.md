# Evals

29 кейсов в YAML, разбитых по агентам. Два способа прогона — быстрый pytest-гейт
и полноценный эксперимент с заливкой в LangFuse для дашборда.

## Три типа проверок (требование диплома)

| Тип | Скорер | Где |
|---|---|---|
| Программный ассерт | `exact_match`, `threshold` | categorization, csv_parsing, security |
| LLM-as-judge | `llm_judge` | categorization (разумна ли категория для мерчанта) |
| Корректность tool-вызова | `tool_call` | tool_routing (supervisor → правильная нода) |

## Кейсы

```
tests/evals/cases/
├── categorization/   # 11 — categorizer (LLM), exact_match + порог confidence + llm_judge
├── cascade/          #  3 — rule→LLM каскад + learning loop (детерминированно)
├── csv_parsing/      #  2 — TinkoffCsvParser (детерминированно)
├── security/         #  3 — injection guard (2 детерм. паттерна + 1 семантический)
├── tool_routing/     #  4 — supervisor routing, tool_call по выбранной ноде
└── multi_intent/     #  6 — Send-веер план (4 exact_match + 2 llm_judge)
```

> `just eval` (pytest-гейт) гоняет все 29 кейсов и даёт полный success rate.
> `just eval-report` (LangFuse) скорит детерминированный субсет 23 кейса
> (categorization 11, cascade*, csv 2, security 3, tool_routing 4); multi_intent
> и async-llm_judge — только в pytest-гейте. *cascade в LangFuse-раннере пока
> падает на конкуренции asyncpg (см. BACKLOG), в pytest-гейте проходит.

Формат кейса — см. `.claude/skills/eval-writer/SKILL.md` и любой `*.yaml`.

> `llm_judge` — асинхронный (зовёт LLM), поэтому работает только в pytest-гейте
> (`just eval`); в `experiment.py` он пропускается, скоры в LangFuse считаются по
> детерминированным скорерам.

## Прогон

```bash
just eval              # pytest -m eval: pass/fail локально (часть кейсов зовёт LLM)
just eval-report       # заливает кейсы в LangFuse datasets + scored experiment
just eval-report security csv_parsing   # только выбранные агенты (без затрат на LLM)
```

`just eval-report` (`tests/evals/experiment.py`) создаёт по датасету на агента
(`ff-<agent>`), прогоняет `dataset.run_experiment(...)` и пишет скоры — каждый
прогон виден в LangFuse UI как Dataset Run с разбивкой по агентам. Нужен поднятый
`just up` и `LANGFUSE_*` в `.env`. Повторный прогон идемпотентен по item id и
создаёт новый run (для сравнения версий промптов / набора 10+ прогонов).

## Почему два раннера

`pytest -m eval` маркирован и быстрый, но НЕ пишет в LangFuse: OTLP-экспортёр
конфликтует с per-test event-loop у pytest-asyncio. `experiment.py` живёт в одном
долгоживущем цикле, поэтому скоры долетают надёжно.
