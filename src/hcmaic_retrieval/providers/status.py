"""No production channel is silently replaced when a provider is unavailable."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProviderStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    available: bool
    reason: str | None = None
    device: str | None = None

    @model_validator(mode="after")
    def require_reason_when_unavailable(self) -> ProviderStatus:
        if not self.available and not (self.reason or "").strip():
            raise ValueError("unavailable provider requires a reason")
        return self


def require_available(status: ProviderStatus) -> None:
    if not status.available:
        raise RuntimeError(status.reason or f"provider {status.name} is unavailable")

