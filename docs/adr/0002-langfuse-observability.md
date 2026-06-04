# 0002 — LangFuse для observability и evals

**Status:** Accepted
**Date:** 2026-05-26

## Контекст

Курс прямо упоминает рекомендует LangFuse 

Альтернативы рассматривал: LangSmith, loom

## Решение

**LangFuse self-host** через docker-compose (Postgres + ClickHouse + Redis + MinIO + worker + web).

## Обоснование

Рекомендация курса

Технически Laminar лучше для long-running multi-agent (transcript view). Для дипломного проекта где курс заточен под LangFuse — выбор однозначный: меньше friction, готовые гайды, скриншоты соответствуют требованию.

## Следствия

**+** Скриншоты dashboard напрямую закрывают требование курса
**+** Prompt management (Phase 3+) — LangFuse сильнее Laminar
**+** Self-host локально → данные не уходят наружу

**−** Тяжелее Laminar в развёртывании (6 контейнеров vs 1)
**−** Менее сильный при отладке длинных multi-agent runs — но в Phase 1-2 нагрузка маленькая

## Migration path

Если в будущем нужен Laminar для multi-agent debugging — используем OpenLLMetry instrumentation (vendor-neutral OTel). Меняется только endpoint, переинструментация не нужна.
