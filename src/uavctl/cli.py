import argparse
import json

from .report import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim, margins, step responses, mission and Monte Carlo for both autopilots.")
    parser.add_argument("--runs", type=int, default=100, help="Monte Carlo runs per autopilot")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.runs, figures=not args.no_figures), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
