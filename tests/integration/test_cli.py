from __future__ import annotations

import json

from hcmaic_retrieval.cli.main import main


def test_demo_search_cli_emits_machine_readable_result(capsys) -> None:
    exit_code = main(["demo-search", "red car", "--top-k", "1"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["frame"]["frame_uid"] == "V1:10"
    assert payload["quality_status"] == "UNVALIDATED_ON_HCMAIC"

