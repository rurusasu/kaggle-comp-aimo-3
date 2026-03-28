"""Kaggle CLI wrapper for push, status polling, and score retrieval."""

import csv
import io
import json
import logging
import os
import re
import subprocess
import time

logger = logging.getLogger(__name__)


class KaggleRunner:
    def __init__(self, kernel_id: str, competition: str):
        self.kernel_id = kernel_id
        self.competition = competition

    def push(self, notebook_dir: str, retries: int = 3) -> dict:
        """Push notebook to Kaggle.

        Returns {"success": bool, "version": int|None, "stdout": str, "stderr": str}.
        Parses version number from stdout like "Kernel version 5 successfully pushed."
        """
        for attempt in range(retries):
            result = subprocess.run(
                ["kaggle", "kernels", "push", "-p", notebook_dir],
                capture_output=True,
                text=True,
                env=_utf8_env(),
            )
            if result.returncode == 0:
                version = _parse_version(result.stdout)
                logger.info("Push succeeded: version=%s, %s", version, result.stdout.strip())
                return {"success": True, "version": version, "stdout": result.stdout, "stderr": result.stderr}
            logger.warning("Push attempt %d failed: %s", attempt + 1, result.stderr.strip())
            if attempt < retries - 1:
                time.sleep(10)
        return {"success": False, "version": None, "stdout": result.stdout, "stderr": result.stderr}

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

    def fetch_score_for_version(self, version: int | None, wait: int = 120, max_retries: int = 5) -> float | None:
        """Fetch the public score for a specific kernel version from competition submissions.

        Matches the submission by looking for "Version {version}" in the description field.
        Retries if the submission hasn't appeared yet (Kaggle has propagation delay).
        Falls back to latest submission if version is None.
        """
        for attempt in range(max_retries):
            if wait > 0:
                delay = wait if attempt == 0 else wait // 2
                logger.info("Waiting %ds for score to propagate (attempt %d/%d)...", delay, attempt + 1, max_retries)
                time.sleep(delay)

            result = subprocess.run(
                ["kaggle", "competitions", "submissions", "-c", self.competition, "--csv"],
                capture_output=True,
                text=True,
                env=_utf8_env(),
            )
            if result.returncode != 0:
                logger.error("Failed to fetch submissions: %s", result.stderr)
                continue

            reader = csv.DictReader(io.StringIO(result.stdout))
            rows = list(reader)
            if not rows:
                logger.warning("No submissions found")
                continue

            # Try to match by version number in description
            if version is not None:
                version_pattern = f"Version {version}"
                for row in rows:
                    desc = row.get("description", "")
                    if version_pattern in desc:
                        status = row.get("status", "")
                        if "ERROR" in status.upper():
                            logger.warning("Version %d submission has error status: %s", version, status)
                            return None
                        if "PENDING" in status.upper() or "RUNNING" in status.upper():
                            logger.info("Version %d submission still pending/running, will retry", version)
                            break  # retry after wait
                        try:
                            score = float(row["publicScore"])
                            logger.info("Version %d score: %.4f", version, score)
                            return score
                        except (KeyError, ValueError):
                            logger.warning("Could not parse score for version %d: %s", version, row)
                            return None
                else:
                    # Version not found in submissions list yet
                    logger.info("Version %d not in submissions (attempt %d/%d)", version, attempt + 1, max_retries)
                    continue

            else:
                # Fallback: use latest submission
                latest = rows[0]
                try:
                    return float(latest["publicScore"])
                except (KeyError, ValueError):
                    logger.warning("Could not parse score from latest: %s", latest)
                    return None

        logger.error("Failed to fetch score after %d attempts", max_retries)
        return None

    # Keep backward compatibility
    def fetch_latest_score(self, wait: int = 120) -> float | None:
        """Fetch the latest public score (legacy method, no version matching)."""
        return self.fetch_score_for_version(version=None, wait=wait, max_retries=1)


def _parse_version(stdout: str) -> int | None:
    """Extract version number from push stdout like 'Kernel version 5 successfully pushed.'"""
    match = re.search(r"[Vv]ersion\s+(\d+)", stdout)
    return int(match.group(1)) if match else None


def _utf8_env():
    """Return env dict with PYTHONUTF8=1 for Windows compatibility."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    return env
