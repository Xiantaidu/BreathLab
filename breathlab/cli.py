"""Unified command line entry point dispatching subcommands to the ``main()`` of their modules."""
from __future__ import annotations

import importlib
import sys
from typing import Dict, NamedTuple


class Command(NamedTuple):
    """A dispatchable subcommand (dotted module path exposing a zero-argument ``main()``, plus a one-line help string)."""

    module: str
    help: str


COMMANDS: Dict[str, Command] = {
    "make-splits": Command("breathlab.data.make_splits",
                           "Build train/dev/test splits for all configured corpora."),
    "train": Command("breathlab.train",
                     "Train a breath detector (CRNN or Conformer)."),
    "evaluate": Command("breathlab.evaluate",
                        "Sweep thresholds on dev, then report test metrics."),
    "infer": Command("breathlab.infer",
                     "Detect breaths in wav files; write TextGrid + JSON."),
    "tg-ap": Command("breathlab.tg_add_ap",
                     "Add an AP tier to existing TextGrid files."),
}

_ALIASES = {"eval": "evaluate", "splits": "make-splits", "predict": "infer"}


def _usage() -> str:
    lines = ["usage: python -m breathlab <command> [options]", "", "commands:"]
    width = max(len(name) for name in COMMANDS)
    for name, cmd in COMMANDS.items():
        lines.append(f"  {name:<{width}}  {cmd.help}")
    lines += [
        "",
        "aliases: " + ", ".join(f"{a} -> {b}" for a, b in _ALIASES.items()),
        "",
        "Run `python -m breathlab <command> --help` for the flags of a command.",
        "Any module can also be called directly, e.g. `python -m breathlab.train ...`.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Dispatch a subcommand to its owning module; returns the process exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0
    if args[0] in ("-V", "--version"):
        from breathlab import __version__

        print(f"BreathLab {__version__}")
        return 0

    name = _ALIASES.get(args[0], args[0])
    if name not in COMMANDS:
        print(f"error: unknown command {args[0]!r}\n")
        print(_usage())
        return 2

    cmd = COMMANDS[name]
    module = importlib.import_module(cmd.module)
    # the target module parses sys.argv itself; rewrite it so its argparse sees only its own flags
    sys.argv = [f"breathlab {name}", *args[1:]]
    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
