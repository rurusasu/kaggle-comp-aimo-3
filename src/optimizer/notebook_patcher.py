"""Patch notebook.py parameter constants via regex replacement."""

import re
from pathlib import Path

# Parameters to patch: name -> type converter
_PARAM_TYPES: dict[str, type] = {
    "NUM_SAMPLES": int,
    "NUM_TIR_ROUNDS": int,
    "MAX_NEW_TOKENS": int,
    "TEMPERATURE": float,
    "CODE_TIMEOUT": int,
}


def patch_notebook(notebook_path: Path, params: dict) -> None:
    """Replace parameter constants in notebook.py with new values.

    Matches lines like `NUM_SAMPLES = 8  # comment` and replaces the value
    while preserving the inline comment.
    """
    content = notebook_path.read_text(encoding="utf-8")

    for name, value in params.items():
        if name not in _PARAM_TYPES:
            raise ValueError(f"Unknown parameter: {name}")
        pattern = rf"^({name}\s*=\s*)(\S+)(\s*#.*)?$"
        formatted = str(value) if isinstance(value, int) else f"{round(value, 6)}"
        replacement = rf"\g<1>{formatted}\g<3>"
        content, count = re.subn(pattern, replacement, content, count=1, flags=re.MULTILINE)
        if count == 0:
            raise ValueError(f"Parameter {name} not found in {notebook_path}")

    notebook_path.write_text(content, encoding="utf-8")


def read_current_params(notebook_path: Path) -> dict:
    """Read current parameter values from notebook.py."""
    content = notebook_path.read_text(encoding="utf-8")
    result = {}
    for name, typ in _PARAM_TYPES.items():
        match = re.search(rf"^{name}\s*=\s*(\S+)", content, re.MULTILINE)
        if match:
            result[name] = typ(match.group(1))
    return result
