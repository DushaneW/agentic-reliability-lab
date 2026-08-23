"""A single named, timestamped measurement, used for run/experiment metrics."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class Metric(BaseModel):
    run_id: str
    name: str
    value: float
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
