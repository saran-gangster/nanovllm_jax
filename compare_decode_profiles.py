"""Compare two decode runtime profile summaries."""

from __future__ import annotations

import argparse
import json

from nanovllm_jax.utils.decode_profile_artifacts import compare_decode_profile_summaries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two decode runtime profile summary JSON files.",
    )
    parser.add_argument(
        "--before",
        required=True,
        help="Baseline decode_runtime_profile_summary.json path.",
    )
    parser.add_argument(
        "--after",
        required=True,
        help="Candidate decode_runtime_profile_summary.json path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    comparison = compare_decode_profile_summaries(args.before, args.after)
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
