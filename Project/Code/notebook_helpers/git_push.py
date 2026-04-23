"""GitHub result-push helper used by the Colab notebook."""

from __future__ import annotations

import subprocess
from pathlib import Path

from constants import (
    COLAB_REPOSITORY_DIR,
    GIT_IDENTITY_EMAIL,
    GIT_IDENTITY_NAME,
    GITHUB_FILE_SIZE_LIMIT_BYTES,
    REPOSITORY_URL,
)


def commit_and_push_results(token: str, run_id: int | None = None) -> None:
    """Stage notebook, masks, and outputs, then commit and push to GitHub."""
    repo_dir = Path(COLAB_REPOSITORY_DIR)
    if not repo_dir.exists():
        raise FileNotFoundError("Repository directory not found. Run Cell 1.1 first.")

    def run_git(command: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True)
        if check and result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip())
        return result

    def get_staged_files() -> list[str]:
        raw_output = run_git(["git", "diff", "--cached", "--name-only", "-z"]).stdout
        return [path for path in raw_output.split(chr(0)) if path]

    run_git(["git", "config", "user.name", GIT_IDENTITY_NAME])
    run_git(["git", "config", "user.email", GIT_IDENTITY_EMAIL])
    run_git(["git", "reset", "--quiet"])

    for relative_path in [
        "Project/Code/CV2_Pipeline.ipynb",
        "Project/Code/data/masks",
        "Project/Code/data/outputs",
    ]:
        if (repo_dir / relative_path).exists():
            run_git(["git", "add", relative_path])

    staged_files = get_staged_files()
    large_files = []
    for relative_path in staged_files:
        absolute_path = repo_dir / relative_path
        if absolute_path.exists() and absolute_path.stat().st_size > GITHUB_FILE_SIZE_LIMIT_BYTES:
            large_files.append((relative_path, absolute_path.stat().st_size))

    if large_files:
        print("Refusing to commit files larger than GitHub's 100 MB limit:")
        for relative_path, size_bytes in large_files:
            print(f"  {relative_path} ({size_bytes / 1e6:.1f} MB)")
        raise RuntimeError("Remove or compress the listed files before committing.")

    if not staged_files:
        print("No staged changes. Repository is already up to date.")
        return

    print("Staged files:")
    for relative_path in staged_files:
        print(f"  {relative_path}")

    if not token:
        print("No token provided. Commit and push skipped.")
        return

    commit_message = f"results: add colab run {run_id:03d}" if isinstance(run_id, int) else "results: add colab outputs"
    commit_result = run_git(["git", "commit", "-m", commit_message], check=False)
    if commit_result.returncode != 0:
        raise RuntimeError(commit_result.stderr or commit_result.stdout)

    authenticated_url = REPOSITORY_URL.replace("https://", f"https://{token}@")
    try:
        run_git(["git", "remote", "set-url", "origin", authenticated_url])
        push_result = run_git(["git", "push", "origin", "main"], check=False)
    finally:
        run_git(["git", "remote", "set-url", "origin", REPOSITORY_URL], check=False)

    if push_result.returncode != 0:
        raise RuntimeError(push_result.stderr or push_result.stdout)
    print(f"Committed and pushed: {commit_message}")
