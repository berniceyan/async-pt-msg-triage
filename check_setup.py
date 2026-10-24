#!/usr/bin/env python3
"""Verifies the environment end to end in a few seconds. The one live call is
an urgency screen on a made-up message, about half a cent."""
from __future__ import annotations

import os
import sys


def main() -> int:
    bad = 0

    print("1. python")
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 11)
    print(f"   {'OK ' if ok else 'FAIL'} {v.major}.{v.minor}.{v.micro} (need 3.11+)")
    bad += not ok

    print("2. packages")
    for mod in ["pydantic", "openai", "pyarrow", "huggingface_hub", "pytest"]:
        try:
            __import__(mod); print(f"   OK  {mod}")
        except ImportError:
            print(f"   FAIL {mod} -- pip install -r requirements.txt"); bad += 1

    print("3. api key")
    from src.env import ENV_PATH, api_key, load_env
    load_env()
    key = api_key()
    if key:
        print(f"   OK  {key[:8]}...{key[-4:]}  ({'.env' if ENV_PATH.exists() else 'shell'})")
    else:
        print("   FAIL OPENROUTER_API_KEY not in .env or the environment")
        print("        cp .env.example .env  and paste your key")
        bad += 1

    if key:
        print("4. live model call")
        try:
            from src.judgment import DEFAULT_MODEL as M, screen
            v, meta = screen("I've had a runny nose for two days. No fever.")
            print(f"   {'OK ' if meta['ok'] else 'FAIL'} {M}")
            print(f"        served   {meta.get('model_served')}")
            print(f"        latency  {meta.get('latency_s')}s")
            print(f"        tokens   {meta.get('in_tokens')} in / {meta.get('out_tokens')} out")
            print(f"        cost     ${meta.get('cost_usd')}")
            if v:
                print(f"        screen   {v.verdict}")
            if not meta["ok"]:
                print(f"        error    {meta.get('error')}"); bad += 1
        except Exception as e:
            print(f"   FAIL {e}"); bad += 1

    print(f"\n{'ALL CHECKS PASSED' if not bad else f'{bad} CHECK(S) FAILED'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
