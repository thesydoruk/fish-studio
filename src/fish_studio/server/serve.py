#!/usr/bin/env python3
"""Run the Fish Speech HTTP TTS server."""

from __future__ import annotations

import argparse

import uvicorn

from fish_studio.config import load_config
from fish_studio.server.app import create_app
from fish_studio.server.settings import ServerSettings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--config", default=".env", help="Path to .env")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project = load_config(args.config)
    project.workspace().ensure_layout()
    settings = ServerSettings.from_project(project)
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
