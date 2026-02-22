import asyncio
from collections.abc import Awaitable, Callable, Iterable
from tqdm.asyncio import tqdm_asyncio
from typing import TypeVar


T = TypeVar("T")
R = TypeVar("R")


async def gather_limited(tasks: list[Awaitable[R]], max_concurrency: int | None = None) -> list[R]:
    if max_concurrency is None or max_concurrency <= 0:
        return list(await tqdm_asyncio.gather(*tasks, leave=False))

    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_one(task: Awaitable[R]) -> R:
        async with semaphore:
            return await task

    wrapped = [run_one(task) for task in tasks]
    return list(await tqdm_asyncio.gather(*wrapped, leave=False))


async def to_thread_map(
    func: Callable[[T], R],
    items: Iterable[T],
    max_concurrency: int | None = None
) -> list[R]:
    item_list = list(items)
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency is not None and max_concurrency > 0 else None

    async def run_item(item: T) -> R:
        if semaphore is None:
            return await asyncio.to_thread(func, item)
        async with semaphore:
            return await asyncio.to_thread(func, item)

    tasks = [run_item(item) for item in item_list]
    return list(await tqdm_asyncio.gather(*tasks, leave=False))
