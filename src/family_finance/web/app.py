"""Starlette application for the local finance dashboard."""

from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path

import structlog
from jinja2 import Environment, FileSystemLoader, select_autoescape
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from family_finance.domain import Category, Direction
from family_finance.infrastructure.settings import get_settings
from family_finance.web.agent import ask_agent
from family_finance.web.dashboard import (
    DashboardFilters,
    DashboardView,
    build_dashboard,
    build_detail,
)

logger = structlog.get_logger()

_WEB_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _WEB_DIR / "templates"
_STATIC_DIR = _WEB_DIR / "static"

_templates = Environment(
    loader=FileSystemLoader(_TEMPLATES_DIR),
    autoescape=select_autoescape(("html", "xml")),
)

# Сообщения для 503: наружу — только generic, детали (DSN, трейс) уходят в лог.
_DB_UNAVAILABLE = (
    "Не удалось прочитать данные из Postgres. Проверь, что инфраструктура "
    "запущена через `just up`."
)
_SERVICE_UNAVAILABLE = "Сервис данных временно недоступен."


async def dashboard_page(request: Request) -> Response:
    """Render the main dashboard page."""
    try:
        family_id = _family_id_from_request(request)
        filters = _filters_from_request(request)
    except ValueError as exc:
        return _render_error(title="Некорректные параметры", message=str(exc), status_code=400)

    try:
        dashboard = await build_dashboard(family_id=family_id, filters=filters)
    except ValueError as exc:
        return _render_error(title="Семья не найдена", message=str(exc), status_code=404)
    except Exception:
        logger.exception("dashboard_render_failed")
        return _render_error(
            title="Dashboard временно недоступен",
            message=_DB_UNAVAILABLE,
            status_code=503,
        )

    return HTMLResponse(_render_dashboard(dashboard))


async def dashboard_api(request: Request) -> Response:
    """Return the same dashboard snapshot as JSON."""
    try:
        family_id = _family_id_from_request(request)
        filters = _filters_from_request(request)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    try:
        dashboard = await build_dashboard(family_id=family_id, filters=filters)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception:
        logger.exception("dashboard_api_failed")
        return JSONResponse({"error": _SERVICE_UNAVAILABLE}, status_code=503)
    return JSONResponse(dashboard.as_dict())


async def transactions_api(request: Request) -> Response:
    """Return transaction rows for the selected dashboard bucket."""
    try:
        family_id = _family_id_from_request(request)
        filters = _filters_from_request(request)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    try:
        if family_id is None:
            dashboard = await build_dashboard(filters=filters)
            if dashboard.selected_family_id is None:
                return JSONResponse({"error": "В базе нет семей"}, status_code=404)
            family_id = uuid.UUID(dashboard.selected_family_id)
        detail = await build_detail(
            family_id,
            filters=filters,
            bucket=request.query_params.get("bucket"),
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception:
        logger.exception("transactions_api_failed")
        return JSONResponse({"error": _SERVICE_UNAVAILABLE}, status_code=503)
    return JSONResponse(detail.as_dict())


async def agent_api(request: Request) -> Response:
    """Ask the existing finance agent graph from the web UI."""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Некорректный JSON"}, status_code=400)

    try:
        if not isinstance(payload, dict):
            raise ValueError("Ожидался JSON-объект")
        family_id_raw = payload.get("family_id")
        question = str(payload.get("question") or "")
        if not family_id_raw:
            raise ValueError("family_id обязателен")
        answer = await ask_agent(family_id=uuid.UUID(str(family_id_raw)), question=question)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception:
        logger.exception("agent_api_failed")
        return JSONResponse({"error": _SERVICE_UNAVAILABLE}, status_code=503)
    return JSONResponse(answer.as_dict())


def _render_dashboard(dashboard: DashboardView) -> str:
    template = _templates.get_template("dashboard.html")
    return template.render(
        dashboard=dashboard,
        chart_payload=dashboard.chart_payload(),
    )


def _render_error(*, title: str, message: str, status_code: int) -> HTMLResponse:
    template = _templates.get_template("error.html")
    return HTMLResponse(
        template.render(title=title, message=message),
        status_code=status_code,
    )


def _family_id_from_request(request: Request) -> uuid.UUID | None:
    raw = request.query_params.get("family_id")
    if not raw:
        return None
    try:
        return uuid.UUID(raw)
    except ValueError as exc:
        msg = f"Некорректный family_id: {raw}"
        raise ValueError(msg) from exc


def _filters_from_request(request: Request) -> DashboardFilters:
    params = request.query_params
    return DashboardFilters(
        period=params.get("period") or "this_month",
        start_date=_parse_date(params.get("start_date")),
        end_date=_parse_date(params.get("end_date")),
        category=_parse_category(params.get("category")),
        direction=_parse_direction(params.get("direction")),
        group_by=params.get("group_by") or "category",
    )


def _parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    return date.fromisoformat(raw)


def _parse_category(raw: str | None) -> Category | None:
    if not raw:
        return None
    return Category(raw)


def _parse_direction(raw: str | None) -> Direction:
    if not raw:
        return Direction.EXPENSE
    return Direction(raw)


app = Starlette(
    debug=get_settings().environment == "dev",
    routes=[
        Route("/", dashboard_page, name="dashboard"),
        Route("/api/dashboard", dashboard_api, name="dashboard_api"),
        Route("/api/transactions", transactions_api, name="transactions_api"),
        Route("/api/agent", agent_api, methods=["POST"], name="agent_api"),
        Mount("/static", StaticFiles(directory=_STATIC_DIR), name="static"),
    ],
)
