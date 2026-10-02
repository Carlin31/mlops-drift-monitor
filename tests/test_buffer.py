import asyncio

from app.observability.buffer import DriftBuffer


def test_append_and_snapshot_returns_added_sample():
    buffer = DriftBuffer(maxlen=10)

    asyncio.run(buffer.append({"a": 1.0}))
    snapshot = asyncio.run(buffer.snapshot())

    assert snapshot == [{"a": 1.0}]
    assert len(buffer) == 1


def test_buffer_evicts_oldest_when_maxlen_exceeded():
    buffer = DriftBuffer(maxlen=2)

    async def fill():
        await buffer.append({"v": 1})
        await buffer.append({"v": 2})
        await buffer.append({"v": 3})

    asyncio.run(fill())
    snapshot = asyncio.run(buffer.snapshot())

    assert snapshot == [{"v": 2}, {"v": 3}]
    assert len(buffer) == 2
