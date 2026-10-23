from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath


def relative(path: str) -> str:
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError("Tool paths must be relative and cannot traverse parents")
    return str(p)


def docker(*args: str, timeout: int = 120, stdin: str | None = None) -> str:
    result = subprocess.run(
        ["docker", *args],
        input=stdin,
        text=True,
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Docker {args[0]} failed: {result.stderr[-2000:]}")
    return result.stdout


class DockerSandbox:
    """Only the task filesystem and dedicated memory survive within a session.

    No host mounts, credentials, network, Docker socket, or writable image filesystem.
    Every continuation starts a new container, including the history-retaining control.
    """

    def __init__(
        self, task, source: Path, memory: Path | None = None, resources: dict | None = None
    ):
        resources = resources or {}
        self.cwd = task.cwd
        self.id = "clm-" + uuid.uuid4().hex[:16]
        if self.cwd in {"/", "/memory", "/tmp"} or not self.cwd.startswith("/"):
            raise ValueError("Use a dedicated absolute task working directory")
        try:
            docker(
                "run",
                "-d",
                "--name",
                self.id,
                "--network",
                "none",
                "--read-only",
                "--user",
                "0:0",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                str(resources.get("pids", 128)),
                "--cpus",
                str(resources.get("cpus", 2)),
                "--memory",
                resources.get("memory", "2g"),
                "--tmpfs",
                f"{self.cwd}:rw,exec,size={resources.get('work_size', '2g')}",
                "--tmpfs",
                "/memory:rw,size=64m",
                "--tmpfs",
                "/tmp:rw,exec,size=1g",
                "--workdir",
                self.cwd,
                "--entrypoint",
                "/bin/sh",
                task.image,
                "-c",
                "while :; do sleep 3600; done",
            )
            self._import(source, self.cwd)
            if memory:
                self._import(memory, "/memory")
        except BaseException:
            self.close()
            raise

    def _import(self, source: Path, target: str):
        # Docker's archive API rejects uploads to a read-only rootfs even when the
        # destination is tmpfs. Extract through exec, leaving the rootfs read-only.
        with tempfile.TemporaryFile() as archive:
            with tarfile.open(fileobj=archive, mode="w") as tar:
                tar.add(source, arcname=".")
            archive.seek(0)
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-i",
                    self.id,
                    "tar",
                    "--no-same-owner",
                    "-xf",
                    "-",
                    "-C",
                    target,
                ],
                stdin=archive,
                capture_output=True,
                timeout=120,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(
                "Failed to load sandbox working files: " + result.stderr.decode()[-2000:]
            )

    def close(self):
        subprocess.run(
            ["docker", "rm", "-f", self.id], capture_output=True, timeout=30, check=False
        )

    def _path(self, path: str, memory: bool = False) -> str:
        return ("/memory" if memory else self.cwd) + "/" + relative(path)

    def read(self, path: str, memory: bool = False) -> str:
        limit = 2_000_000 if memory and path == "context.json" else 60000
        return docker("exec", self.id, "head", "-c", str(limit), "--", self._path(path, memory))

    def write(self, path: str, content: str, memory: bool = False):
        target = self._path(path, memory)
        docker("exec", self.id, "mkdir", "-p", str(PurePosixPath(target).parent))
        # argv supplies the filename; stdin supplies content. No shell interpolation.
        docker("exec", "-i", self.id, "sh", "-c", 'cat > "$1"', "write", target, stdin=content)

    def run(self, command: str, timeout: int) -> dict:
        before = time.monotonic()
        # A separate process group is killed on timeout; no surviving shell children.
        code = r"""
import os, signal, subprocess, sys, tempfile
with tempfile.TemporaryFile() as output:
    p = subprocess.Popen(sys.argv[1], shell=True, start_new_session=True,
                         stdout=output, stderr=subprocess.STDOUT)
    try:
        p.wait(timeout=int(sys.argv[2]))
        status = p.returncode
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        status = 124
    output.seek(0)
    print(output.read(60000).decode(errors="replace"))
    print("\nCLM_EXIT_STATUS=" + str(status))
"""
        result = docker(
            "exec", self.id, "python3", "-c", code, command, str(timeout), timeout=timeout + 20
        )
        output, marker, status = result.rpartition("\nCLM_EXIT_STATUS=")
        if not marker:
            raise RuntimeError("Sandbox command wrapper failed")
        return {"exit_code": int(status), "output": output, "seconds": time.monotonic() - before}

    def list_files(self):
        code = "import os,json; print(json.dumps(sorted(os.listdir('.'))[:256]))"
        return json.loads(docker("exec", self.id, "python3", "-c", code))

    def snapshot(self, destination: Path, memory_destination: Path | None = None):
        self._export(self.cwd, destination)
        if memory_destination:
            self._export("/memory", memory_destination)

    def _export(self, source: str, destination: Path):
        # Read mounted tmpfs through the running process. Docker cp can export the
        # underlying empty image directory instead of its mounted contents.
        destination.mkdir(parents=True, exist_ok=False)
        with tempfile.TemporaryFile() as archive:
            result = subprocess.run(
                ["docker", "exec", self.id, "tar", "-cf", "-", "-C", source, "."],
                stdout=archive,
                stderr=subprocess.PIPE,
                timeout=120,
                check=False,
            )
            if result.returncode:
                raise RuntimeError("Sandbox snapshot failed: " + result.stderr.decode()[-2000:])
            archive.seek(0)
            with tarfile.open(fileobj=archive, mode="r:") as tar:
                for member in tar:
                    path = PurePosixPath(member.name)
                    if (
                        path.is_absolute()
                        or ".." in path.parts
                        or not (member.isfile() or member.isdir())
                    ):
                        raise ValueError("Snapshot contains a link, special file, or escaping path")
                    target = destination / str(path)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with tar.extractfile(member) as src, target.open("wb") as out:
                            shutil.copyfileobj(src, out)
                        target.chmod(member.mode & 0o777)


class FixtureSandbox:
    """File-only offline sandbox. Cannot execute model-generated code on the host."""

    def __init__(self, task, source: Path, memory: Path | None = None, resources=None):
        self.temp = tempfile.TemporaryDirectory(prefix="clm-fixture-")
        self.root = Path(self.temp.name)
        shutil.copytree(source, self.root / "work")
        if memory:
            shutil.copytree(memory, self.root / "memory")
        else:
            (self.root / "memory").mkdir()

    def _path(self, path, memory=False):
        root = self.root / ("memory" if memory else "work")
        p = root / relative(path)
        if not p.resolve().is_relative_to(root.resolve()):
            raise ValueError("Escaping symlink")
        return p

    def read(self, path, memory=False):
        limit = 2_000_000 if memory and path == "context.json" else 60000
        with self._path(path, memory).open() as f:
            return f.read(limit)

    def write(self, path, content, memory=False):
        p = self._path(path, memory)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def run(self, command, timeout):
        raise ValueError("The offline fixture has no shell; use Docker for real models")

    def list_files(self):
        return sorted(p.name for p in (self.root / "work").iterdir())[:256]

    def snapshot(self, destination, memory_destination=None):
        shutil.copytree(self.root / "work", destination)
        if memory_destination:
            shutil.copytree(self.root / "memory", memory_destination)

    def close(self):
        self.temp.cleanup()


def make_sandbox(task, source, memory=None, *, fixture=False, resources=None):
    cls = FixtureSandbox if fixture else DockerSandbox
    return cls(task, source, memory, resources)


def file_manifest(root: Path) -> dict:
    import hashlib

    result = {}
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            p = Path(base) / name
            if p.is_symlink():
                raise ValueError(f"Symlink artifacts are not permitted: {p.relative_to(root)}")
        for name in files:
            p = Path(base) / name
            if not p.is_file():
                raise ValueError("Only regular files are permitted")
            result[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result
