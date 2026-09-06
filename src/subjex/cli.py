"""Command-line interface for SubjEx."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .classifier import DEFAULT_MODEL, classify_subjects, get_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="subjex",
        description="Extract and classify grammatical subjects in English text.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="UTF-8 input file. If omitted, text is read from standard input.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output TSV file. If omitted, results are written to standard output.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"spaCy pipeline to use (default: {DEFAULT_MODEL}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the SubjEx command-line interface."""

    args = build_parser().parse_args(argv)
    text = args.input.read_text(encoding="utf-8") if args.input else sys.stdin.read()

    try:
        pipeline = get_pipeline(args.model)
    except OSError:
        print(
            f"spaCy model '{args.model}' is not installed. Run:\n"
            f"  python -m spacy download {args.model}",
            file=sys.stderr,
        )
        return 2

    result = classify_subjects(text, pipeline=pipeline)
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    elif result:
        print(result)
    return 0
