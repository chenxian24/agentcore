"""AgentCore CLI entry point.

Usage:
    python -m agentcore              — print version and available commands
    python -m agentcore serve        — start the web API server
    python -m agentcore version      — print version
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentcore", description="AgentCore — Atomic agent engine")
    sub = parser.add_subparsers(dest="command")

    sub.add_command = sub.add_parser  # type: ignore[attr-defined]

    version_p = sub.add_parser("version", help="Print version")
    version_p.set_defaults(cmd="version")

    serve_p = sub.add_parser("serve", help="Start the web API server")
    serve_p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    serve_p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    serve_p.set_defaults(cmd="serve")

    args = parser.parse_args()

    if not args.command:
        from agentcore import __version__
        print(f"agentcore {__version__}")
        parser.print_help()
        return

    if args.cmd == "version":
        from agentcore import __version__
        print(f"agentcore {__version__}")
        return

    if args.cmd == "serve":
        try:
            import uvicorn
        except ImportError:
            print("Error: uvicorn not installed. Install with: pip install agentcore[web]", file=sys.stderr)
            sys.exit(1)
        uvicorn.run("agentcore.web.app:app", host=args.host, port=args.port, reload=False)
        return


if __name__ == "__main__":
    main()
