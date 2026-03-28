from unittest.mock import MagicMock, patch

from src.optimizer.kaggle_runner import KaggleRunner, _parse_version


def test_parse_version_from_push_output():
    assert _parse_version("Kernel version 5 successfully pushed.") == 5
    assert _parse_version("Kernel version 117 successfully pushed.  Please check...") == 117
    assert _parse_version("Some other output") is None


@patch("src.optimizer.kaggle_runner.subprocess.run")
def test_push_returns_success_with_version(mock_run):
    mock_run.return_value = MagicMock(
        returncode=0, stdout="Kernel version 5 successfully pushed.", stderr=""
    )
    runner = KaggleRunner(kernel_id="user/my-kernel", competition="my-comp")
    result = runner.push("kaggle-notebook/")
    assert result["success"] is True
    assert result["version"] == 5
    mock_run.assert_called_once()


@patch("src.optimizer.kaggle_runner.subprocess.run")
def test_push_failure(mock_run):
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="409 Conflict")
    runner = KaggleRunner(kernel_id="user/my-kernel", competition="my-comp")
    result = runner.push("kaggle-notebook/")
    assert result["success"] is False
    assert result["version"] is None


@patch("src.optimizer.kaggle_runner.subprocess.run")
def test_get_status_complete(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout='{"status": "complete"}', stderr="")
    runner = KaggleRunner(kernel_id="user/my-kernel", competition="my-comp")
    assert runner.get_status() == "complete"


@patch("src.optimizer.kaggle_runner.subprocess.run")
def test_get_status_running(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout='{"status": "running"}', stderr="")
    runner = KaggleRunner(kernel_id="user/my-kernel", competition="my-comp")
    assert runner.get_status() == "running"


@patch("src.optimizer.kaggle_runner.subprocess.run")
def test_fetch_score_for_version_match(mock_run):
    submissions_output = (
        "fileName,date,description,status,publicScore,privateScore\n"
        "submission.csv,2026-03-28,Notebook My Kernel | Version 5,complete,0.45,0.0\n"
        "submission.csv,2026-03-27,Notebook My Kernel | Version 4,complete,0.30,0.0\n"
    )
    mock_run.return_value = MagicMock(returncode=0, stdout=submissions_output, stderr="")
    runner = KaggleRunner(kernel_id="user/my-kernel", competition="my-comp")
    score = runner.fetch_score_for_version(version=5, wait=0, max_retries=1)
    assert score == 0.45


@patch("src.optimizer.kaggle_runner.subprocess.run")
def test_fetch_score_for_version_picks_correct_version(mock_run):
    submissions_output = (
        "fileName,date,description,status,publicScore,privateScore\n"
        "submission.csv,2026-03-28,Notebook My Kernel | Version 5,complete,0.45,0.0\n"
        "submission.csv,2026-03-27,Notebook My Kernel | Version 4,complete,0.30,0.0\n"
    )
    mock_run.return_value = MagicMock(returncode=0, stdout=submissions_output, stderr="")
    runner = KaggleRunner(kernel_id="user/my-kernel", competition="my-comp")
    score = runner.fetch_score_for_version(version=4, wait=0, max_retries=1)
    assert score == 0.30


@patch("src.optimizer.kaggle_runner.subprocess.run")
def test_fetch_score_for_version_error_status(mock_run):
    submissions_output = (
        "fileName,date,description,status,publicScore,privateScore\n"
        "submission.csv,2026-03-28,Notebook My Kernel | Version 5,SubmissionStatus.ERROR,,\n"
    )
    mock_run.return_value = MagicMock(returncode=0, stdout=submissions_output, stderr="")
    runner = KaggleRunner(kernel_id="user/my-kernel", competition="my-comp")
    score = runner.fetch_score_for_version(version=5, wait=0, max_retries=1)
    assert score is None


@patch("src.optimizer.kaggle_runner.subprocess.run")
def test_fetch_score_for_version_not_found(mock_run):
    submissions_output = (
        "fileName,date,description,status,publicScore,privateScore\n"
        "submission.csv,2026-03-27,Notebook My Kernel | Version 4,complete,0.30,0.0\n"
    )
    mock_run.return_value = MagicMock(returncode=0, stdout=submissions_output, stderr="")
    runner = KaggleRunner(kernel_id="user/my-kernel", competition="my-comp")
    score = runner.fetch_score_for_version(version=99, wait=0, max_retries=1)
    assert score is None


@patch("src.optimizer.kaggle_runner.subprocess.run")
def test_fetch_latest_score_fallback(mock_run):
    submissions_output = (
        "fileName,date,description,status,publicScore,privateScore\n"
        "submission.csv,2026-03-28,auto,complete,0.45,0.0\n"
    )
    mock_run.return_value = MagicMock(returncode=0, stdout=submissions_output, stderr="")
    runner = KaggleRunner(kernel_id="user/my-kernel", competition="my-comp")
    score = runner.fetch_latest_score(wait=0)
    assert score == 0.45


@patch("src.optimizer.kaggle_runner.subprocess.run")
def test_fetch_latest_score_no_submissions(mock_run):
    mock_run.return_value = MagicMock(
        returncode=0, stdout="fileName,date,description,status,publicScore,privateScore\n", stderr=""
    )
    runner = KaggleRunner(kernel_id="user/my-kernel", competition="my-comp")
    score = runner.fetch_latest_score(wait=0)
    assert score is None
