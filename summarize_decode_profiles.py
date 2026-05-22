"""Aggregate multiple decode runtime profile summaries."""

from __future__ import annotations

import argparse
import json

from nanovllm_jax.utils.decode_profile_artifacts import summarize_decode_profile_runs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate multiple decode runtime profile summary JSON files.",
    )
    parser.add_argument(
        "summaries",
        nargs="+",
        help="One or more decode_runtime_profile_summary.json paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = summarize_decode_profile_runs(args.summaries)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
