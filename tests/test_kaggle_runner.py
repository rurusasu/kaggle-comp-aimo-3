from unittest.mock import MagicMock, patch

from src.optimizer.kaggle_runner import KaggleRunner


@patch("src.optimizer.kaggle_runner.subprocess.run")
def test_push_returns_success(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout="Kernel version 5 pushed.", stderr="")
    runner = KaggleRunner(kernel_id="user/my-kernel", competition="my-comp")
    result = runner.push("kaggle-notebook/")
    assert result["success"] is True
    mock_run.assert_called_once()


@patch("src.optimizer.kaggle_runner.subprocess.run")
def test_push_failure_raises(mock_run):
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="409 Conflict")
    runner = KaggleRunner(kernel_id="user/my-kernel", competition="my-comp")
    result = runner.push("kaggle-notebook/")
    assert result["success"] is False


@patch("src.optimizer.kaggle_runner.subprocess.run")
def test_get_status_complete(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout='{"status": "complete"}', stderr="")
    runner = KaggleRunner(kernel_id="user/my-kernel", competition="my-comp")
    status = runner.get_status()
    assert status == "complete"


@patch("src.optimizer.kaggle_runner.subprocess.run")
def test_get_status_running(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout='{"status": "running"}', stderr="")
    runner = KaggleRunner(kernel_id="user/my-kernel", competition="my-comp")
    status = runner.get_status()
    assert status == "running"


@patch("src.optimizer.kaggle_runner.subprocess.run")
def test_fetch_latest_score(mock_run):
    submissions_output = (
        "fileName,date,description,status,publicScore,privateScore\n"
        "submission.csv,2026-03-27,auto,complete,0.45,0.0\n"
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
