from __future__ import annotations

import json

from hcmaic_retrieval.cli.main import main


def test_demo_search_cli_emits_machine_readable_result(capsys) -> None:
    exit_code = main(["demo-search", "red car", "--top-k", "1"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["frame"]["frame_uid"] == "V1:10"
    assert payload["quality_status"] == "UNVALIDATED_ON_HCMAIC"


def test_demo_trake_and_qa_commands_are_runnable(capsys) -> None:
    assert main(["demo-trake", "person enters, then person leaves"]) == 0
    trake = json.loads(capsys.readouterr().out)
    assert trake["sequences"][0]["video_id"] == "V1"

    assert main(["demo-qa", "What car is shown?"]) == 0
    qa = json.loads(capsys.readouterr().out)
    assert qa["needs_human_review"] is True
    assert qa["evidence"]
