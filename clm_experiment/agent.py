from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .tasks import evaluate, search
from .types import Cost, Grade, Task, as_json, dump

SYSTEM = """You are a Context Language Model working in an editable task environment.
The task and this system prompt are pinned. Your remaining conversation is mirrored
as a JSON list of {role, content} messages in /memory/context.json before each action.
You may freely edit that file with run, or use the context tool to replace its list.
Only user and assistant roles with string content are accepted; keep enough information
to work within the context budget. No automatic summary is provided on overflow.

Return exactly one JSON object per turn, without Markdown fences. Tools:
{"tool":"list"}
{"tool":"read","path":"relative/file","memory":false}
{"tool":"write","path":"relative/file","content":"...","memory":false}
{"tool":"run","command":"shell command inside the isolated work container"}
{"tool":"context","messages":[{"role":"user","content":"retained context"}]}
{"tool":"search","query":"fixed-corpus search query"}
{"tool":"evaluate"}
{"tool":"finish"}

All conditions have exactly these tools. search is available only for BrowseComp-Plus.
The run command uses the task working directory. memory=true accesses /memory, which
holds context, handoffs, and notes. Functional artifacts belong in the working directory.
Do not put transcripts, summaries, reasoning logs, or disguised memory in functional
artifacts. Executable code, tests, structured program inputs, and results may persist.
Fresh continuations receive the original task and functional artifacts only.
"""


@dataclass
class State:
    history: list[dict] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)

    def message(self, role: str, content: str, origin: str):
        self.history.append({"role": role, "content": content})
        self.transcript.append({"role": role, "content": content, "origin": origin})


def validate_history(value) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError("Context must be a message list")
    result = []
    for m in value:
        if not isinstance(m, dict) or m.get("role") not in {"user", "assistant"}:
            raise ValueError("Invalid context role")
        if not isinstance(m.get("content"), str):
            raise ValueError("Context content must be text")
        result.append({"role": m["role"], "content": m["content"]})
    return result


def messages(task: Task, state: State, budget: int) -> list[dict]:
    # Consecutive same-role turns are merged for Anthropic Messages compatibility.
    seq = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task.prompt + f"\nContext budget: {budget} tokens."},
    ]
    for msg in state.history:
        if seq[-1]["role"] == msg["role"]:
            seq[-1] = {"role": msg["role"], "content": seq[-1]["content"] + "\n" + msg["content"]}
        else:
            seq.append(dict(msg))
    # The API must end on a user turn, even after a context edit.
    if seq[-1]["role"] != "user":
        seq.append({"role": "user", "content": "Take the next action."})
    return seq


class Session:
    def __init__(self, task, sandbox, provider, config, state=None):
        self.task, self.sandbox, self.provider = task, sandbox, provider
        self.config = config
        self.state = state or State()
        self.cost = Cost()
        self.grade_seconds = 0.0
        self.evaluations = 0
        self.retrieved_docids: set[str] = set()

    def grade(self) -> Grade:
        before = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="clm-grade-") as tmp:
            p = Path(tmp) / "work"
            self.sandbox.snapshot(p)
            from .sandbox import file_manifest

            file_manifest(p)
            result = evaluate(self.task, p)
        self.grade_seconds += time.monotonic() - before
        self.evaluations += 1
        return result

    def _tool(self, action: dict) -> dict:
        tool = action["tool"]
        if tool == "read":
            content = self.sandbox.read(action["path"], action.get("memory", False))[:60000]
            self.cost.inspection_calls += 1
            self.cost.inspection_bytes += len(content.encode())
            return {"path": action["path"], "content": content}
        if tool == "write":
            if not isinstance(action["content"], str):
                raise ValueError("File content must be text")
            self.sandbox.write(action["path"], action["content"], action.get("memory", False))
            return {"written": action["path"]}
        if tool == "list":
            result = self.sandbox.list_files()
            self.cost.inspection_calls += 1
            self.cost.inspection_bytes += len(json.dumps(result).encode())
            return {"files": result}
        if tool == "run":
            result = self.sandbox.run(action["command"], self.config["command_timeout"])
            # Count all shell calls/output as inspection as well: labels cannot game read costs.
            self.cost.inspection_calls += 1
            self.cost.inspection_bytes += len(result["output"].encode())
            return result
        if tool == "context":
            value = validate_history(action["messages"])
            self.sandbox.write("context.json", json.dumps(value), memory=True)
            return {"context_edit": "submitted"}
        if tool == "search":
            if self.task.suite != "browsecomp_plus":
                raise ValueError("No fixed-corpus search configured for this task")
            docs = search(self.task.evaluator["corpus_db"], action["query"])
            self.retrieved_docids.update(d["docid"] for d in docs)
            self.cost.inspection_calls += 1
            self.cost.inspection_bytes += len(json.dumps(docs).encode())
            return {"documents": docs}
        if tool == "evaluate":
            grade = self.grade()
            # No gold answers, relevance labels, evaluator paths, or hidden reports in observations.
            return {"valid": grade.valid, "score": grade.score}
        if tool == "finish":
            return {"finished": True}
        raise ValueError(f"Unknown tool: {tool}")

    def step(self, seed: int) -> str:
        budget = self.config["context_tokens"]
        seq = messages(self.task, self.state, budget)
        count = self.provider.count_tokens(seq)
        if count > budget - self.config["context_reserve"]:
            return "context_exhausted"
        if count > 0.65 * budget:
            self.state.message(
                "user",
                "Context is above 65% of budget. Compact "
                "/memory/context.json if needed before the next request.",
                "harness",
            )
            seq = messages(self.task, self.state, budget)
            if self.provider.count_tokens(seq) > budget - self.config["context_reserve"]:
                return "context_exhausted"
        mirror = json.dumps(self.state.history)
        self.sandbox.write("context.json", mirror, memory=True)
        reply = self.provider.complete(seq, seed)
        self.cost.add(reply.cost)
        self.state.message("assistant", reply.text, "model")
        started = time.monotonic()
        tool = "invalid"
        try:
            action = json.loads(reply.text)
            if not isinstance(action, dict):
                raise ValueError("Expected one JSON action")
            tool = action.get("tool", "invalid")
            observation = self._tool(action)
            changed = self.sandbox.read("context.json", memory=True)
            if changed != mirror:
                edited = validate_history(json.loads(changed))
                candidate = State(history=edited)
                if self.provider.count_tokens(messages(self.task, candidate, budget)) <= (
                    budget - self.config["context_reserve"]
                ):
                    self.state.history = edited
                    observation["context_edit_accepted"] = True
                else:
                    observation["context_edit_accepted"] = False
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            observation = {"tool_error": str(exc)[:2000]}
        self.cost.tool_calls += 1
        self.cost.tool_seconds += time.monotonic() - started
        self.state.message("user", json.dumps(observation), "tool")
        return "agent_finished" if tool == "finish" else "continue"

    def segment(
        self, calls: int, seed: int, target: float | None = None, cost_limit: float | None = None
    ) -> dict:
        started = time.monotonic()
        status = "budget_exhausted"
        grade = self.grade()
        if target is not None and self.task.reached(grade, target):
            status = "target_already_reached"
        else:
            for i in range(calls):
                if (
                    cost_limit is not None
                    and self.cost.units(self.config["cost_weights"]) >= cost_limit
                ):
                    break
                status = self.step(seed + i)
                if (i + 1) % self.config["grade_every"] == 0 or status != "continue":
                    grade = self.grade()
                    if target is not None and self.task.reached(grade, target):
                        status = "success"
                        break
                if status != "continue":
                    break
            else:
                status = "budget_exhausted"
            grade = self.grade()
            if target is not None and self.task.reached(grade, target):
                status = "success"
            elif status == "continue":
                status = "budget_exhausted"
        self.cost.wall_seconds += time.monotonic() - started
        return {
            "status": status,
            "grade": as_json(grade),
            "cost": as_json(self.cost),
            "evaluations": self.evaluations,
            "grade_seconds": self.grade_seconds,
            "retrieved_docids": sorted(self.retrieved_docids),
        }

    def save(self, directory: Path, result: dict):
        dump(directory / "result.json", result)
        dump(directory / "transcript.json", self.state.transcript)
        dump(directory / "context.json", self.state.history)
