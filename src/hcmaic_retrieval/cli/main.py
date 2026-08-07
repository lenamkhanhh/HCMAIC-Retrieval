"""CLI for validation, demo task execution, and local serving."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from hcmaic_retrieval.artifacts.kaggle import validate_kaggle_bundle
from hcmaic_retrieval.config import load_config
from hcmaic_retrieval.contracts import KISQuery
from hcmaic_retrieval.service.api import create_app
from hcmaic_retrieval.service.runtime import build_batch1_runtime, build_demo_runtime
from hcmaic_retrieval.tasks.qa import QAQuery
from hcmaic_retrieval.tasks.trake import TRAKEQuery


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hcmaic")
    commands = parser.add_subparsers(dest="command", required=True)

    demo_search = commands.add_parser("demo-search")
    demo_search.add_argument("query")
    demo_search.add_argument("--top-k", type=int, default=10)

    demo_trake = commands.add_parser("demo-trake")
    demo_trake.add_argument("query")
    demo_trake.add_argument("--top-k", type=int, default=5)

    demo_qa = commands.add_parser("demo-qa")
    demo_qa.add_argument("question")

    validate = commands.add_parser("validate-artifacts")
    validate.add_argument("--config", type=Path, required=True)

    serve = commands.add_parser("serve")
    serve.add_argument("--config", type=Path)
    serve.add_argument("--device", default="cpu")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-artifacts":
        config = load_config(args.config)
        report = validate_kaggle_bundle(config.dataset)
        _print(report.model_dump(mode="json", exclude={"catalog"}))
        return 0 if report.ok else 2

    if args.command == "serve":
        import uvicorn

        runtime = (
            build_batch1_runtime(load_config(args.config), device=args.device)
            if args.config
            else build_demo_runtime()
        )
        uvicorn.run(create_app(runtime), host=args.host, port=args.port)
        return 0

    runtime = build_demo_runtime()
    response: BaseModel
    if args.command == "demo-search":
        response = runtime.search_kis(
            KISQuery(
                query_id="demo-kis",
                task="TKIS",
                text=args.query,
                top_k=args.top_k,
            )
        )
    elif args.command == "demo-trake":
        response = runtime.search_trake(
            TRAKEQuery(query_id="demo-trake", text=args.query, top_k=args.top_k)
        )
    elif args.command == "demo-qa":
        response = runtime.answer_qa(QAQuery(query_id="demo-qa", question=args.question))
    else:  # pragma: no cover - argparse prevents this branch
        raise AssertionError(args.command)
    _print(
        {
            **response.model_dump(mode="json"),
            "quality_status": runtime.quality_status,
        }
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
