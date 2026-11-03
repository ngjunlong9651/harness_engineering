"""A minimal agent harness, small enough to read in one sitting.

Agent = Model + Harness. This file is the harness: everything except the model.

    tools        what the agent can do                  (Tool, make_tools)
    sandbox      where it is allowed to do it           (Sandbox)
    policy       what it may do, and for how long       (Policy)
    context      what the model gets to see each step   (Context, with compaction)
    loop         think -> act -> observe -> repeat      (run_agent)
    memory       state that survives across sessions    (progress log, see notebook)
    models       a scripted stand-in for an LLM         (ScriptedBugfixModel)

The model here is scripted so the notebook runs offline and deterministically.
Swap in a real LLM by implementing the same one-method interface:
    model.next_action(messages, tools) -> {"type": "tool_use", "name": ..., "args": {...}}
                                          or {"type": "final", "text": ...}
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


# --------------------------------------------------------------------------- tools
@dataclass
class Tool:
    """A capability the model can request. The schema is what the model sees; fn is what the harness runs."""
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., str]
    risk: str = "read"          # "read" | "write" | "exec": the policy decides which risks are allowed


class Sandbox:
    """Confines file access to one directory. The model never gets raw filesystem access."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def resolve(self, rel: str) -> Path:
        # Resolve symlinks and ".." first, THEN check containment. Checking the raw string is a classic bug.
        p = (self.root / rel).resolve()
        if self.root != p and self.root not in p.parents:
            raise PermissionError(f"path escapes sandbox: {rel}")
        return p


def make_tools(sb: Sandbox) -> dict[str, Tool]:
    def list_files() -> str:
        return "\n".join(sorted(str(p.relative_to(sb.root)) for p in sb.root.rglob("*") if p.is_file()))

    def read_file(path: str) -> str:
        return sb.resolve(path).read_text()

    def write_file(path: str, content: str) -> str:
        sb.resolve(path).write_text(content)
        return f"wrote {len(content)} chars to {path}"

    def run_tests() -> str:
        # Verification is a tool: the agent gets objective feedback instead of guessing it worked.
        # -B: do not write .pyc files. Without it, a fast edit that keeps the file size and mtime can leave a stale
        # bytecode cache, and the tests would keep running the OLD code. Harness bugs are often environment bugs.
        r = subprocess.run([sys.executable, "-B", "-m", "unittest", "-q"], cwd=sb.root,
                           capture_output=True, text=True, timeout=30)
        return f"exit={r.returncode}\n{r.stdout}{r.stderr}".strip()

    def run_shell(command: str) -> str:
        # Deliberately dangerous. The default policy denies the "exec" risk class.
        return subprocess.run(command, shell=True, cwd=sb.root, capture_output=True, text=True).stdout

    obj = lambda **props: {"type": "object", "properties": props, "required": list(props)}
    s = {"type": "string"}
    return {t.name: t for t in [
        Tool("list_files", "List all files in the workspace.", obj(), list_files, "read"),
        Tool("read_file", "Read a text file from the workspace.", obj(path=s), read_file, "read"),
        Tool("write_file", "Overwrite a file in the workspace.", obj(path=s, content=s), write_file, "write"),
        Tool("run_tests", "Run the project's unit tests and return the output.", obj(), run_tests, "exec_safe"),
        Tool("run_shell", "Run an arbitrary shell command.", obj(command=s), run_shell, "exec"),
    ]}


# -------------------------------------------------------------------------- policy
@dataclass
class Policy:
    """Rules the harness enforces no matter what the model asks for."""
    allowed_risks: set = field(default_factory=lambda: {"read", "write", "exec_safe"})
    max_steps: int = 12                 # hard stop: agents that loop forever are a real failure mode
    max_observation_chars: int = 2000   # cap tool output so one huge file cannot flood the context

    def allows(self, tool: Tool) -> bool:
        return tool.risk in self.allowed_risks


# ------------------------------------------------------------------------- context
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)       # rough rule of thumb: about 4 characters per token


class Context:
    """The model's working memory. Everything the model knows this step is in `messages`.

    Context is finite and every token costs money and attention, so the harness actively manages it.
    """

    def __init__(self, budget_tokens: int = 100_000, strategy: str = "none", keep_recent: int = 4):
        self.messages: list[dict] = []
        self.budget = budget_tokens
        self.strategy = strategy          # "none" | "mask" | "summarize"
        self.keep_recent = keep_recent
        self.compactions = 0

    def add(self, role: str, content: str, **extra):
        self.messages.append({"role": role, "content": content, **extra})

    def tokens(self) -> int:
        return sum(estimate_tokens(str(m.get("content", "")) + str(m.get("args", ""))) for m in self.messages)

    def maybe_compact(self):
        if self.strategy == "none" or self.tokens() <= self.budget:
            return
        self.compactions += 1
        head, tail = self.messages[:-self.keep_recent], self.messages[-self.keep_recent:]
        if self.strategy == "mask":
            # Observation masking: keep what the agent DID, drop the bulky results it already used.
            for m in head:
                if m["role"] == "tool" and not m.get("masked"):
                    m["content"] = f"[output of {m.get('name')} elided, {len(m['content'])} chars]"
                    m["masked"] = True
        elif self.strategy == "summarize":
            # Compaction: replace the old head with a short summary (a real harness would ask an LLM).
            calls = [m["name"] for m in head if m["role"] == "assistant" and m.get("type") == "tool_use"]
            summary = f"[summary of earlier work: {len(calls)} tool calls ({', '.join(sorted(set(calls)))})]"
            first_user = self.messages[0]
            self.messages = [first_user, {"role": "user", "content": summary}] + tail


# --------------------------------------------------------------------------- loop
def run_agent(model, task: str, tools: dict[str, Tool], policy: Policy, ctx: Context, verbose=True) -> dict:
    """The whole idea of an agent in a for-loop: ask the model, run its tool, show it the result."""
    ctx.add("user", task)
    trace = []
    for step in range(1, policy.max_steps + 1):
        ctx.maybe_compact()
        action = model.next_action(ctx.messages, tools)

        if action["type"] == "final":                       # model says it is done
            trace.append({"step": step, "event": "final", "detail": action["text"], "tokens": ctx.tokens()})
            if verbose: print(f"[{step}] FINAL: {action['text']}")
            return {"status": "done", "answer": action["text"], "trace": trace, "context": ctx}

        name, args = action["name"], action.get("args", {})
        ctx.add("assistant", f"call {name}", type="tool_use", name=name, args=args)
        tool = tools.get(name)
        if tool is None:                                    # models hallucinate tools; say so, do not crash
            obs, event = f"ERROR: unknown tool '{name}'. Available: {sorted(tools)}", "unknown_tool"
        elif not policy.allows(tool):                       # the policy, not the model, has the last word
            obs, event = f"DENIED: tool '{name}' has risk '{tool.risk}', which this policy does not allow.", "denied"
        else:
            try:
                obs, event = str(tool.fn(**args)), "ok"
            except Exception as e:                          # errors become observations the model can react to
                obs, event = f"ERROR: {type(e).__name__}: {e}", "error"

        if len(obs) > policy.max_observation_chars:         # protect the context window
            obs = obs[:policy.max_observation_chars] + f"\n[truncated {len(obs) - policy.max_observation_chars} chars]"
        ctx.add("tool", obs, name=name)
        trace.append({"step": step, "event": event, "tool": name, "args": args, "tokens": ctx.tokens(),
                      "detail": obs.splitlines()[0][:80] if obs else ""})
        if verbose: print(f"[{step}] {event.upper():12} {name}({args if len(str(args)) < 60 else '...'}) -> {trace[-1]['detail']}")

    return {"status": "max_steps", "answer": None, "trace": trace, "context": ctx}


# --------------------------------------------------------------- scripted "model"
class ScriptedBugfixModel:
    """A stand-in for an LLM that fixes the bug in calc.py. It decides from what it has SEEN, like a real model.

    try_dangerous=True makes it attempt a destructive shell command first, to show the policy at work.
    """

    def __init__(self, try_dangerous: bool = False):
        self.try_dangerous = try_dangerous

    def next_action(self, messages, tools):
        tool_calls = [m for m in messages if m["role"] == "assistant" and m.get("type") == "tool_use"]
        results = [m for m in messages if m["role"] == "tool"]
        last = results[-1] if results else None

        if self.try_dangerous and not tool_calls:
            return {"type": "tool_use", "name": "run_shell", "args": {"command": "rm -rf /"}}
        if last is None or last["name"] == "run_shell":
            return {"type": "tool_use", "name": "list_files", "args": {}}
        if last["name"] == "list_files":
            return {"type": "tool_use", "name": "read_file", "args": {"path": "calc.py"}}
        if last["name"] == "read_file":
            return {"type": "tool_use", "name": "run_tests", "args": {}}
        if last["name"] == "run_tests":
            if "exit=0" in last["content"]:
                return {"type": "final", "text": "All tests pass. Fixed add() to use + instead of -."}
            source = next(m for m in reversed(results) if m["name"] == "read_file")["content"]
            fixed = re.sub(r"return a - b(\s*# add)?", "return a + b", source, count=1)
            return {"type": "tool_use", "name": "write_file", "args": {"path": "calc.py", "content": fixed}}
        if last["name"] == "write_file":
            return {"type": "tool_use", "name": "run_tests", "args": {}}
        return {"type": "final", "text": "I am stuck."}


class LoopyModel:
    """Reads the same large file over and over. Used to show why context management matters."""

    def next_action(self, messages, tools):
        n = sum(1 for m in messages if m["role"] == "assistant" and m.get("type") == "tool_use")
        if n >= 14:
            return {"type": "final", "text": "done reading"}
        return {"type": "tool_use", "name": "read_file", "args": {"path": "big.txt"}}


# ------------------------------------------------------------------- workspaces
def make_bugfix_workspace(root: Path | None = None) -> Path:
    """A tiny project with one failing test, created in a temp directory."""
    root = Path(root or tempfile.mkdtemp(prefix="harness_ws_"))
    (root / "calc.py").write_text(textwrap.dedent('''\
        def add(a, b):
            return a - b  # BUG: should add

        def mul(a, b):
            return a * b
        '''))
    (root / "test_calc.py").write_text(textwrap.dedent('''\
        import unittest
        from calc import add, mul

        class T(unittest.TestCase):
            def test_add(self):
                self.assertEqual(add(2, 3), 5)
            def test_mul(self):
                self.assertEqual(mul(2, 3), 6)

        if __name__ == "__main__":
            unittest.main()
        '''))
    return root
