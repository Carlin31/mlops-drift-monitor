import asyncio
from collections import deque


class DriftBuffer:
    def __init__(self, maxlen: int) -> None:
        self._buffer: deque[dict[str, float]] = deque(maxlen=maxlen)
        self._lock = asyncio.Lock()

    async def append(self, sample: dict[str, float]) -> None:
        async with self._lock:
            self._buffer.append(sample)

    async def snapshot(self) -> list[dict[str, float]]:
        async with self._lock:
            return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)
