# 0001 — LangGraph для оркестрации multi-agent

**Status:** Accepted
**Date:** 2026-05-26

## Контекст

Курс будет учить supervisor pattern 1 июня — нужен фреймворк где этот паттерн родной.

## Решение

**LangGraph 1.2.0+**.

## Обоснование


## Mitigation

- Используем `create_react_agent` для простых нод (cut boilerplate)
- StateGraph для главного графа (полный контроль)
- LangChain memory не используем — он deprecated в 2026; только LangGraph checkpointer
