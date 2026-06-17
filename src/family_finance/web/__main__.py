"""Run the local web dashboard."""

from __future__ import annotations

import uvicorn


def main() -> None:
    """Start the dashboard on localhost."""
    uvicorn.run("family_finance.web.app:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
