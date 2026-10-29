"""
Start the Bank Marketing dashboard (Flask API + HTML/CSS/JS frontend).

    python run.py
    python run.py --port 8080
    python run.py --host 0.0.0.0

Requires the trained artifacts in models/ and outputs/ (they ship with the
project; regenerate with `python train_and_evaluate.py`).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.app import create_app


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the Bank Marketing dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true", help="Flask debug mode (auto-reload)")
    args = parser.parse_args(argv)

    print("Loading data and models (once)...", flush=True)
    app = create_app()
    print(f"Ready in {app.config['SERVICE'].startup_seconds}s. "
          f"Open http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}",
          flush=True)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
