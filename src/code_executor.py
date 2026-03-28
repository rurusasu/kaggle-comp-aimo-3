"""Safe Python code execution for Tool-Integrated Reasoning (TIR).

Extracts Python code blocks from model output, executes them in a
restricted environment with timeout, and returns the output.
"""

import multiprocessing
import re
import signal
import sys
import traceback
from io import StringIO


# Maximum execution time per code block (seconds)
CODE_TIMEOUT = 30

# Maximum output length (characters)
MAX_OUTPUT_LEN = 4096

# Allowed modules for math computation
ALLOWED_MODULES = {
    "math", "cmath", "decimal", "fractions", "itertools", "functools",
    "collections", "operator", "string", "re", "random", "statistics",
    "sympy", "numpy", "scipy",
}

# Blocked patterns (dangerous operations)
BLOCKED_PATTERNS = [
    r"\bimport\s+os\b", r"\bimport\s+sys\b", r"\bimport\s+subprocess\b",
    r"\bimport\s+shutil\b", r"\bimport\s+pathlib\b",
    r"\b__import__\b", r"\beval\b", r"\bexec\b",
    r"\bopen\s*\(", r"\bos\.", r"\bsys\.",
    r"\bsubprocess\.", r"\bshutil\.",
]


def extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks from model output.

    Supports:
    - ```python ... ```
    - ```py ... ```
    - ``` ... ```  (when it looks like Python)
    """
    blocks = []

    # Pattern: ```python or ```py
    for match in re.finditer(r"```(?:python|py)\s*\n(.*?)```", text, re.DOTALL):
        blocks.append(match.group(1).strip())

    # Pattern: ``` (generic) — only if no python-tagged blocks found
    if not blocks:
        for match in re.finditer(r"```\s*\n(.*?)```", text, re.DOTALL):
            code = match.group(1).strip()
            # Heuristic: looks like Python if it has common Python patterns
            if any(kw in code for kw in ("print(", "def ", "for ", "import ", "=", "range(")):
                blocks.append(code)

    return blocks


def is_code_safe(code: str) -> tuple[bool, str]:
    """Check if code is safe to execute."""
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, code):
            return False, f"Blocked pattern: {pattern}"
    return True, ""


def _execute_in_process(code: str, result_queue: multiprocessing.Queue):
    """Execute code in a subprocess (called by multiprocessing)."""
    stdout_capture = StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_capture

    try:
        # Restricted globals
        safe_globals = {"__builtins__": __builtins__}

        # Pre-import allowed modules
        for mod_name in ["math", "itertools", "functools", "collections", "fractions", "decimal"]:
            try:
                safe_globals[mod_name] = __import__(mod_name)
            except ImportError:
                pass

        # Try importing sympy/numpy (may not be available)
        for mod_name in ["sympy", "numpy"]:
            try:
                safe_globals[mod_name] = __import__(mod_name)
                if mod_name == "numpy":
                    safe_globals["np"] = safe_globals["numpy"]
                if mod_name == "sympy":
                    safe_globals["sp"] = safe_globals["sympy"]
            except ImportError:
                pass

        exec(code, safe_globals)
        output = stdout_capture.getvalue()
        result_queue.put(("success", output[:MAX_OUTPUT_LEN]))
    except Exception:
        output = stdout_capture.getvalue()
        tb = traceback.format_exc()
        result_queue.put(("error", (output + "\n" + tb)[:MAX_OUTPUT_LEN]))
    finally:
        sys.stdout = old_stdout


def execute_code(code: str, timeout: int = CODE_TIMEOUT) -> tuple[bool, str]:
    """Execute Python code safely with timeout.

    Returns:
        (success: bool, output: str)
    """
    # Safety check
    safe, reason = is_code_safe(code)
    if not safe:
        return False, f"Code blocked: {reason}"

    try:
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_execute_in_process,
            args=(code, result_queue),
        )
        process.start()
        process.join(timeout=timeout)

        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
            return False, f"Execution timed out after {timeout}s"

        if result_queue.empty():
            return False, "No output (process may have crashed)"

        status, output = result_queue.get_nowait()
        return status == "success", output

    except Exception as e:
        return False, f"Execution error: {e}"


def execute_code_simple(code: str, timeout: int = CODE_TIMEOUT) -> tuple[bool, str]:
    """Simple code execution without multiprocessing (for environments where fork is unavailable).

    Uses exec() directly with a timeout signal (Unix only) or no timeout (Windows).
    """
    safe, reason = is_code_safe(code)
    if not safe:
        return False, f"Code blocked: {reason}"

    stdout_capture = StringIO()
    old_stdout = sys.stdout

    # Set up globals
    safe_globals = {"__builtins__": __builtins__}
    for mod_name in ["math", "itertools", "functools", "collections", "fractions", "decimal"]:
        try:
            safe_globals[mod_name] = __import__(mod_name)
        except ImportError:
            pass
    for mod_name in ["sympy", "numpy"]:
        try:
            safe_globals[mod_name] = __import__(mod_name)
            if mod_name == "numpy":
                safe_globals["np"] = safe_globals["numpy"]
            if mod_name == "sympy":
                safe_globals["sp"] = safe_globals["sympy"]
        except ImportError:
            pass

    try:
        sys.stdout = stdout_capture

        # Timeout via signal (Unix only)
        has_alarm = hasattr(signal, "SIGALRM")
        if has_alarm:
            def handler(signum, frame):
                raise TimeoutError(f"Execution timed out after {timeout}s")
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)

        try:
            exec(code, safe_globals)
            output = stdout_capture.getvalue()
            return True, output[:MAX_OUTPUT_LEN]
        except TimeoutError as e:
            return False, str(e)
        except Exception:
            output = stdout_capture.getvalue()
            tb = traceback.format_exc()
            return False, (output + "\n" + tb)[:MAX_OUTPUT_LEN]
        finally:
            if has_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
    finally:
        sys.stdout = old_stdout
