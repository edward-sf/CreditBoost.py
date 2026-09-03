import pytest


@pytest.mark.slow
def test_search_cli_handles_unmeasurable_candidates(fixture_path, capsys):
    """search_cli succeeds and reports when no candidate can be scored.

    On small datasets, no group reaches MIN_FAIRNESS_GROUP_SIZE, so every
    candidate fails to score. search_cli prints the entire frontier with
    failed_reasons and then returns 0, communicating that it completed its job
    (inspection succeeded, ranking completed) even though nothing was selectable.
    """
    from creditboost.search_cli import main

    code = main(["--data", str(fixture_path)])
    assert code == 0

    captured = capsys.readouterr()
    assert "no candidate could be scored; nothing could be selected" in captured.out
    # Verify the frontier was printed before the message
    assert "candidate" in captured.out
    assert "AUC" in captured.out
