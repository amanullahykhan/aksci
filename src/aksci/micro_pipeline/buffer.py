"""
Micro-pipeline: memory-bounded, chunked data flow.

The core idea is simple and deliberately not magic: never materialize more
than `chunk_size` rows/items in memory between stages. Data is streamed
through Python generators, and large files are read using pandas' native
`chunksize` reader (or Polars' lazy/streaming API), so a file far bigger
than available RAM can still be processed safely.

This is honest about its limits: it is a single-process, generator-based
pipeline. It reduces memory pressure and keeps code simple; it does not
parallelize CPU-bound work across cores by itself. For that, wrap a stage's
function with `concurrent.futures.ProcessPoolExecutor` if a stage is truly
CPU-bound and independent per chunk.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Callable, Deque, Iterable, Iterator, List, Optional

import pandas as pd


class BoundedBuffer:
    """A fixed-capacity FIFO buffer.

    Once `maxsize` is reached, pushing a new item drops the oldest one
    rather than growing unboundedly -- this guarantees memory use for the
    buffer itself never exceeds `maxsize` items, independent of how much
    data flows through it overall.
    """

    def __init__(self, maxsize: int = 1000) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be a positive integer")
        self.maxsize = maxsize
        self._queue: Deque[Any] = deque()

    def push(self, item: Any) -> None:
        if len(self._queue) >= self.maxsize:
            self._queue.popleft()
        self._queue.append(item)

    def __len__(self) -> int:
        return len(self._queue)

    def drain(self) -> Iterator[Any]:
        """Yield and remove all currently buffered items, oldest first."""
        while self._queue:
            yield self._queue.popleft()


class Stage:
    """One unit of work in a micro-pipeline: a named, pure transformation."""

    def __init__(self, name: str, func: Callable[[Any], Any]) -> None:
        self.name = name
        self.func = func

    def run(self, stream: Iterable[Any]) -> Iterator[Any]:
        for item in stream:
            yield self.func(item)


class MicroPipeline:
    """Chains `Stage` objects and streams data through them chunk-by-chunk.

    Example
    -------
    >>> pipeline = MicroPipeline(chunk_size=50_000)
    >>> pipeline.add_stage("drop_nulls", lambda df: df.dropna())
    >>> pipeline.add_stage("add_margin", lambda df: df.assign(
    ...     margin=df["revenue"] - df["cost"]))
    >>> for processed_chunk in pipeline.run_csv("sales.csv"):
    ...     processed_chunk.to_parquet("out.parquet", engine="pyarrow", append=True)
    """

    def __init__(self, chunk_size: int = 10_000) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self.chunk_size = chunk_size
        self._stages: List[Stage] = []

    def add_stage(self, name: str, func: Callable[[Any], Any]) -> "MicroPipeline":
        """Append a stage. Returns self so calls can be chained."""
        self._stages.append(Stage(name, func))
        return self

    def run(self, source: Iterable[Any]) -> Iterator[Any]:
        """Run an in-memory iterable of items through all stages, lazily."""
        stream: Iterable[Any] = source
        for stage in self._stages:
            stream = stage.run(stream)
        yield from stream

    def run_csv(self, path: str, **read_csv_kwargs: Any) -> Iterator[pd.DataFrame]:
        """Stream a CSV file through the pipeline in bounded-size chunks.

        Uses pandas' native chunked reader (`chunksize=self.chunk_size`),
        so files much larger than available RAM can be processed: at any
        moment only one chunk's worth of rows is in memory.
        """
        reader = pd.read_csv(path, chunksize=self.chunk_size, **read_csv_kwargs)
        for chunk in reader:
            processed: Any = chunk
            for stage in self._stages:
                processed = stage.func(processed)
            yield processed

    def run_polars_lazy(self, path: str) -> "Any":
        """Stream a CSV via Polars' lazy/streaming engine for very large files.

        Returns a LazyFrame with stage functions applied as `.map_batches`
        transformations, evaluated only when `.collect(streaming=True)` is
        called -- Polars manages the memory-bounded execution internally.
        """
        import polars as pl  # local import: optional dependency

        lazy = pl.scan_csv(path)
        for stage in self._stages:
            lazy = lazy.map_batches(stage.func)
        return lazy
