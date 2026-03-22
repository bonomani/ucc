from __future__ import annotations

import argparse
import json
import sys

from .engine import UccMvpEngine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the UCC MVP reference engine.")
    parser.add_argument("declaration", nargs="?", help="Path to a declaration message JSON file. Defaults to stdin.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the result JSON.")
    args = parser.parse_args(argv)

    try:
        if args.declaration:
            with open(args.declaration, "r", encoding="utf-8") as handle:
                message = json.load(handle)
        else:
            message = json.load(sys.stdin)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"invalid input: {exc}", file=sys.stderr)
        return 2

    result = UccMvpEngine().execute(message)
    json.dump(result, sys.stdout, indent=2 if args.pretty else None)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
