#!/usr/bin/env python3
"""Run all operator tests (assignment task 2.8 / CI)."""
import os
import subprocess
import sys

OPS = [
    "add",
    "argmax",
    "embedding",
    "linear",
    "rms_norm",
    "rope",
    "self_attention",
    "swiglu",
]


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    device = "cpu"
    args = sys.argv[1:]
    if "--device" in args:
        i = args.index("--device")
        if i + 1 < len(args):
            device = args[i + 1]

    for name in OPS:
        script = os.path.join(root, "ops", f"{name}.py")
        print(f"\n======== {name} ========")
        subprocess.check_call([sys.executable, script, "--device", device])

    print("\n\033[92mAll operator tests passed!\033[0m\n")


if __name__ == "__main__":
    main()
