"""Kaggle CLI wrapper for push, status polling, and score retrieval."""

import csv
import io
import json
import logging
import os
import subprocess
import time

logger = logging.getLogger(__name__)


class KaggleRunner:
    def __init__(self, kernel_id: str, competition: str):
        self.kernel_id = kernel_id
        self.competition = competition

    def push(self, notebook_dir: str, retries: int = 3) -> dict:
        """Push notebook to Kaggle. Returns {"success": bool, "stdout": str, "stderr": str}."""
        for attempt in range(retries):
            result = subprocess.run(
                ["kaggle", "kernels", "push", "-p", notebook_dir],
                capture_output=True,
                text=True,
                env=_utf8_env(),
            )
            if result.returncode == 0:
                logger.info("Push succeeded: %s", result.stdout.strip())
                return {"success": True, "stdout": result.stdout, "stderr": result.stderr}
            logger.warning("Push attempt %d failed: %s", attempt + 1, result.stderr.strip())
            if attempt < retries - 1:
                time.sleep(10)
        return {"success": False, "stdout": result.stdout, "stderr": result.stderr}

    def get_status(self) -> str:
        """Get kernel execution status. Returns one of: complete, running, error, cancelled, queued."""
        result = subprocess.run(
            ["kaggle", "kernels", "status", self.kernel_id],
            capture_output=True,
            text=True,
            env=_utf8_env(),
        )
        if result.returncode != 0:
            logger.error("Status check failed: %s", result.stderr)
            return "error"
        try:
            data = json.loads(result.stdout)
            return data.get("status", "unknown")
        except json.JSONDecodeError:
            stdout = result.stdout.lower()
            for s in ("complete", "running", "error", "cancelled", "queued"):
                if s in stdout:
                    return s
            return "unknown"

    def poll_until_complete(self, timeout: int = 21600, interval: int = 300) -> str:
        """Poll kernel status until terminal state or timeout. Returns final status string."""
        start = time.time()
        while time.time() - start < timeout:
            status = self.get_status()
            logger.info("Kernel status: %s (%.0fs elapsed)", status, time.time() - start)
            if status in ("complete", "error", "cancelled"):
                return status
            time.sleep(interval)
        logger.warning("Polling timed out after %ds", timeout)
        return "timeout"

    def fetch_latest_score(self, wait: int = 120) -> float | None:
        """Fetch the latest public score from competition submissions.

        Waits `wait` seconds for score to propagate, then parses CSV output.
        Returns score as float, or None if not available.
        """
        if wait > 0:
            logger.info("Waiting %ds for score to propagate...", wait)
            time.sleep(wait)

        result = subprocess.run(
            ["kaggle", "competitions", "submissions", "-c", self.competition, "--csv"],
            capture_output=True,
            text=True,
            env=_utf8_env(),
        )
        if result.returncode != 0:
            logger.error("Failed to fetch submissions: %s", result.stderr)
            return None

        reader = csv.DictReader(io.StringIO(result.stdout))
        rows = list(reader)
        if not rows:
            return None

        latest = rows[0]
        try:
            return float(latest["publicScore"])
        except (KeyError, ValueError):
            logger.warning("Could not parse score from: %s", latest)
            return None


def _utf8_env():
    """Return env dict with PYTHONUTF8=1 for Windows compatibility."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    return env
