"""Allow ``python -m breathlab <command>`` to dispatch to :mod:`breathlab.cli`."""
from __future__ import annotations

from breathlab.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
