"""Standard-library-only release selection, provenance and output protection."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import uuid


ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def release_paths(root=ROOT):
    """An allowlist, never a recursive copy of the working directory."""
    root = Path(root).resolve()
    specification = read_json(root / "release_files.json")
    selected = set()
    forbidden = {".git", ".tmp", "__pycache__", "external", "assignment_materials",
                 "results", "evidence", "release_candidates"}
    for pattern in specification["include"]:
        matches = [p for p in root.glob(pattern) if p.is_file()]
        if not matches:
            raise FileNotFoundError(f"Release pattern has no files: {pattern}")
        for path in matches:
            relative = path.relative_to(root)
            if (relative.parts[0] in forbidden or relative.parts[0].startswith(".venv")
                    or any(part in {".git", "__pycache__"} for part in relative.parts)):
                raise ValueError(f"Excluded release path: {relative}")
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Release path escapes project: {relative}")
            selected.add(relative.as_posix())
    return sorted(selected)


def execution_paths(root=ROOT):
    return [name for name in release_paths(root)
            if name.endswith((".py", ".cmd", ".ps1"))
            or name.startswith(("configs/", "scenes/"))
            or name in {"requirements.txt", "resources.lock.json", "robot_assets.lock.json",
                        "release_files.json"}]


def new_output(parent, prefix=""):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = Path(parent).resolve() / (prefix + run_id)
    output.mkdir(parents=True, exist_ok=False)
    return output


def git_state(root=ROOT):
    executable = shutil.which("git")
    fallback = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe"
    executable = executable or (str(fallback) if fallback.is_file() else None)
    if executable is None:
        return {"commit": None, "status": None, "reason": "Git executable unavailable"}
    # Git writes UTF-8, while Windows text=True may decode using CP936.
    top = subprocess.run([executable, "-C", str(root), "-c", "core.quotepath=false",
                          "rev-parse", "--show-toplevel"], capture_output=True)
    if top.returncode or Path(top.stdout.decode("utf-8").strip()).resolve() != Path(root).resolve():
        return {"commit": None, "status": None, "reason": "No Git repository at this project root"}
    revision = subprocess.run([executable, "-C", str(root), "rev-parse", "--verify", "HEAD"],
                              capture_output=True)
    status = subprocess.run([executable, "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                            capture_output=True)
    return {"commit": revision.stdout.decode("utf-8").strip() if revision.returncode == 0 else None,
            "status": status.stdout.decode("utf-8").strip() if status.returncode == 0 else None,
            "reason": None if revision.returncode == 0 else "No HEAD commit yet"}


def verify_assets(root=ROOT):
    """Verify model bytes without requiring a Git checkout at runtime."""
    lock = read_json(Path(root) / "robot_assets.lock.json")
    resource = read_json(Path(root) / "resources.lock.json")["robotis"]
    if lock["resource_commit"] != resource["commit"]:
        raise ValueError("Resource lock and asset lock disagree")
    directory = Path(root) / resource["directory"] / "robotis_tb3"
    for relative, expected in lock["files_sha256"].items():
        path = directory / relative
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"Missing/changed official asset: {relative}. Run scripts/setup.ps1.")
    return {"resource_commit": lock["resource_commit"],
            "verified_asset_files": len(lock["files_sha256"])}


def save_manifest(folder):
    folder = Path(folder)
    write_json(folder / "manifest.json", {"files_sha256": {
        p.relative_to(folder).as_posix(): digest(p) for p in sorted(folder.rglob("*"))
        if p.is_file() and p != folder / "manifest.json"}})


def project_python_sources(root=ROOT):
    """Active modules only; exclude immutable results and evidence trees."""
    root = Path(root)
    return sorted([*root.glob("*.py"), *root.glob("evaluation/**/*.py")])
