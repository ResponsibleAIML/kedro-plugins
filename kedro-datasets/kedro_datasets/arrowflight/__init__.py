"""``AbstractDataSet`` implementations that reads from Arrow pyarrow flight."""

__all__ = ["ArrowFlightDataSet"]

from contextlib import suppress

with suppress(ImportError):
    from .flight_dataset import ArrowFlightDataSet
