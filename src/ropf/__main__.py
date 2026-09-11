"""Entry point for ``python -m ropf``, equivalent to the ``ropf`` console script."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
