from __future__ import annotations

import pytest

from hcmaic_retrieval.providers import ProviderStatus, require_available


def test_provider_status_reports_explicit_disabled_reason() -> None:
    status = ProviderStatus(
        name="asr",
        version="not-built",
        available=False,
        reason="ASR artifact is absent",
    )

    assert status.available is False
    assert status.reason == "ASR artifact is absent"


def test_require_available_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="ASR artifact is absent"):
        require_available(
            ProviderStatus(
                name="asr",
                version="not-built",
                available=False,
                reason="ASR artifact is absent",
            )
        )

