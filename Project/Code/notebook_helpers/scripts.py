"""Script execution helpers for notebook cells."""

from __future__ import annotations

import subprocess
import sys


def run_python_script(script_path: str, *args: str) -> None:
    """Run a project Python script with streamed output."""
    command = [sys.executable, "-u", script_path, *args]
    recent_output = []
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is not None:
        for output_line in process.stdout:
            recent_output.append(output_line.rstrip())
            recent_output = recent_output[-80:]
            print(output_line, end="", flush=True)
    process.wait()
    if process.returncode != 0:
        tail = "\n".join(recent_output[-40:]) if recent_output else "No subprocess output was captured."
        raise RuntimeError(
            f"Script failed with exit code {process.returncode}: {' '.join(command)}\n"
            f"Last captured output:\n{tail}"
        )
