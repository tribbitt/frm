"""What BARC code may do, read off its source without running it.

Standard library only, no package-relative imports: the sandbox worker (``_worker.py``, under the sandbox venv)
imports this file by path and re-checks every source before it executes it.

    check(source)          None if the source may run in the sandbox, else the reason
    rule_colours(source)   the colours the rule names (`Color.X` outside generate_input)
    header(source)         the "# concepts:" and "# description:" comment blocks
    normalised(source)     the code without comments, docstrings or formatting (ast.dump)

The check is the first of several layers (sandbox.py lists them all; the OS sandbox and the
worker's audit hook are what holds if this list misses a route).  It refuses imports other
than ALLOWED_IMPORTS; the builtins and module names that execute, import, open or reflect
(BANNED_NAMES); any attribute that starts with an underscore (dunders and privates: the routes
from an object to its class, frame or globals, `random._os`), that names a module reachable
from an allowed one (`typing.sys`, `typing.operator`), a file or process operation, or a frame
(BANNED_ATTRS, BANNED_ATTR_PREFIXES); string constants holding a dunder (format-string
traversal); and assignment to an attribute chain rooted at an imported name (`np.random.seed =
...`, `R.seed = ...` after `from numpy import random as R`): every program in a worker shares
numpy, so one that rebinds it would change the next one's verdicts.  A store through a plain
variable (`m = np; m.sum = f`) is past what a static check can follow; the worker's runtime
comparison of the shared modules catches it.
"""

from __future__ import annotations

import ast
import re
import sys

ALLOWED_IMPORTS = frozenset({"numpy", "typing", "math", "random", "itertools", "collections", "common"})
BANNED_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "getattr", "setattr", "delattr", "globals", "locals",
    "vars", "dir", "breakpoint", "input", "help", "exit", "quit", "memoryview", "__builtins__", "__loader__",
    "__spec__", "__file__", "__build_class__", "os", "sys", "subprocess", "socket", "shutil", "pathlib", "io",
    "importlib", "ctypes", "pickle", "marshal", "gc", "inspect", "builtins", "signal", "resource", "threading",
    "multiprocessing", "asyncio", "tempfile", "glob", "urllib", "http", "requests", "pdb", "code", "codeop"})
BANNED_ATTRS = frozenset({
    # modules reachable as attributes of allowed ones (typing.sys, typing.operator, np.compat.os, ...)
    "os", "sys", "modules", "builtins", "subprocess", "importlib", "ctypes", "ctypeslib", "socket", "shutil",
    "pathlib", "io", "pickle", "marshal", "gc", "inspect", "operator", "types", "functools", "contextlib",
    "warnings", "stdlib_re", "re", "copyreg", "weakref", "linecache", "tokenize", "traceback", "atexit",
    "platform", "threading", "signal", "compat", "core", "f2py", "distutils", "testing", "lib",
    # process, file and reflection operations
    "system", "popen", "spawn", "fork", "kill", "exec", "execv", "execve", "eval", "compile", "open", "load",
    "loads", "loadtxt", "save", "savez", "savez_compressed", "savetxt", "fromfile", "tofile", "memmap",
    "genfromtxt", "fromregex", "DataSource", "dump", "dumps", "lookfor", "show_config", "show_runtime",
    "getattr", "setattr", "delattr", "globals", "locals", "vars", "unlink", "rmdir", "rename", "chmod", "chdir",
    "mkdir", "write_text", "write_bytes", "read_text", "read_bytes", "environ", "getenv", "putenv"})
BANNED_ATTR_PREFIXES = ("_", "f_", "tb_", "gi_", "cr_", "ag_", "co_")
MODULE_NAMES = frozenset({"np", "numpy", "random", "math", "itertools", "collections", "typing", "common", "Color"})
MAX_SOURCE = 64_000

COLOURS = {"BLACK": 0, "BLUE": 1, "RED": 2, "GREEN": 3, "YELLOW": 4, "GREY": 5, "GRAY": 5, "PINK": 6,
           "ORANGE": 7, "TEAL": 8, "MAROON": 9, "TRANSPARENT": 0, "BACKGROUND": 0}


def _root(name: str) -> str:
    return name.split(".")[0]


def check(source: str) -> str | None:
    """None if `source` passes the static policy, else why it does not."""
    if len(source) > MAX_SOURCE:
        return "too_long"
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return "syntax"
    aliases = set(MODULE_NAMES)

    def bad_path(dotted: str) -> bool:
        return any(p.startswith("_") or p in BANNED_ATTRS for p in dotted.split(".")[1:])

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if _root(a.name) not in ALLOWED_IMPORTS or bad_path(a.name):
                    return f"import:{a.name}"
                aliases.add(a.asname or _root(a.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module or _root(node.module) not in ALLOWED_IMPORTS or bad_path(node.module):
                return f"import:{node.module}"
            for a in node.names:
                if a.name.startswith("_") or a.name in BANNED_ATTRS:
                    return f"import_name:{a.name}"
                if a.name != "*":
                    aliases.add(a.asname or a.name)      # `from numpy import random as R`
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id in BANNED_NAMES:
                return f"name:{node.id}"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith(BANNED_ATTR_PREFIXES) or node.attr in BANNED_ATTRS:
                return f"attr:{node.attr}"
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                root = node.value
                while isinstance(root, ast.Attribute):
                    root = root.value                    # np.random.seed = ...: rooted at np
                if isinstance(root, ast.Name) and root.id in aliases:
                    return f"module_store:{root.id}.{node.attr}"
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            if any(n in aliases or n in BANNED_NAMES for n in node.names):
                return "global_module"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value \
                and node.value != "__main__":
            return "dunder_string"
    if not any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body):
        return "no_main"
    return None


def rule_colours(source: str) -> list[int]:
    """The colours the rule names: every `Color.X` outside `generate_input` (which draws the
    inputs and is not part of the rule), black included (the audit removes the rule's background)."""
    tree = ast.parse(source)
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "generate_input":
            skip |= {id(n) for n in ast.walk(node)}
    out = set()
    for node in ast.walk(tree):
        if id(node) in skip or not isinstance(node, ast.Attribute):
            continue
        if isinstance(node.value, ast.Name) and node.value.id == "Color" and node.attr in COLOURS:
            out.add(COLOURS[node.attr])
    return sorted(out)


def _block(source: str, key: str) -> str:
    """The comment block that starts with `# key:`, joined into one line."""
    lines, on = [], False
    for raw in source.splitlines():
        s = raw.strip()
        if re.match(rf"#\s*{key}\s*:", s, re.IGNORECASE):
            on = True
            rest = s.split(":", 1)[1].strip()
            if rest:
                lines.append(rest)
            continue
        if on:
            if s.startswith("#") and not re.match(r"#\s*\w+\s*:\s*$", s):
                lines.append(s.lstrip("#").strip())
            else:
                break
    return " ".join(x for x in lines if x)


def header(source: str) -> dict:
    return {"concepts": _block(source, "concepts"), "description": _block(source, "description")}


def norm_text(s: str) -> str:
    """Lower case, letters and digits, single spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s.lower())).strip()


class _Strip(ast.NodeTransformer):
    """Drops docstrings and the `if __name__ == "__main__":` block."""

    @staticmethod
    def _body(body):
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            body = body[1:]
        return body or [ast.Pass()]

    def visit_Module(self, node):
        self.generic_visit(node)
        node.body = [n for n in self._body(node.body) if not (
            isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
            and n.test.left.id == "__name__")]
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.body = self._body(node.body)
        return node

    visit_ClassDef = visit_FunctionDef


# 3.13 stopped printing empty fields by default; show_empty=True restores 3.12's text, so code hashes agree on
# 3.12+ (3.11's AST has no type_params fields: its hashes differ, though they stay consistent within one build)
_DUMP = {"show_empty": True} if sys.version_info >= (3, 13) else {}


def normalised(source: str) -> str:
    """The program without comments, docstrings, formatting or its test block."""
    return ast.dump(_Strip().visit(ast.parse(source)), annotate_fields=False, **_DUMP)
