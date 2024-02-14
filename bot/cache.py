from __future__ import annotations

import functools
from asyncio import Queue, create_task
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from itertools import groupby
from typing import Optional, AsyncIterator, Iterable, Generic, TypeVar, Callable, Any, Deque, List, Awaitable

from discord import Message, Guild, Thread, User, Reaction, CategoryChannel, StageChannel, ForumChannel, VoiceChannel, \
    TextChannel, Member
from discord.abc import Messageable, GuildChannel
from discord.mixins import Hashable
from discord.utils import get, find


def queue(method):
    async def _queue(self, *args, **kwargs) -> None:
        self.put_nowait(lambda: method(self, *args, **kwargs))

    return _queue

# TODO Python must have better ways of doing this
class FunctionQueue(Queue):

    def add_dynamic_worker(self, func: Callable[[], Awaitable[Any]]):
        async def task():
            try:
                while True:
                    await func()
            except Exception as e:
                print(f"Task failed with error: {e}")
                raise

        self.workers.append(create_task(task()))
    def add_worker(self):
        async def do_task():
            try:
                func = await self.get()
                await func()

                self.task_done()
            except Exception as e:
                print(f"Task failed with error: {e}")

        self.add_dynamic_worker(do_task)

    def __init__(self):
        super().__init__()
        self.workers = deque()
        self.exec = None

    async def dump_exec(self):
        for i in range(3):
            self.add_worker()

        self.exec = create_task(self.join())
        await self.exec

        self.cancel()

    def done(self) -> bool:
        return len(self.workers) == 0 and self.empty()
    def cancel(self) -> None:
        for worker in self.workers:
            worker.cancel()

        if self.exec is not None: self.exec.cancel()

        self.exec = None
        self.workers.clear()
        self._init(0) # clears the queue


TObject = TypeVar('TObject')
TTarget = TypeVar('TTarget')


# TODO; No tuple unpacking (lambda (message, reactions):
class CacheEntry(Generic[TObject]):

    def __init__(self, current: TObject):
        self.current = current

    async def live(self) -> TObject:
        raise NotImplementedError

    # TODO: Just this for now until we have a better idea of what to do with an offline mirror\
    @property
    def id(self) -> int | str:
        return self.current if hasattr(self.current, 'id') else self._dumb_id_gen()
    def _dumb_id_gen(self) -> str:
        if isinstance(self.current, Reaction): return f'{self.current.message.id}:{self.current}'
        raise NotImplementedError(f'for {type(self.current)}')
    def __eq__(self, other: object) -> bool:
        if isinstance(other, CacheEntry): return self.id == other.id
        return self.id == CacheEntry(current=other).id  # TODO Might need to check object type here, but probably not
    
    # TODO: Just isolate to this until we know what to do
    def is_reaction(self) -> bool: return isinstance(self.current, Reaction)
    def is_message(self) -> bool: return isinstance(self.current, Message)
    def is_guild(self) -> bool: return isinstance(self.current, Guild)
    def is_channel(self) -> bool: return isinstance(self.current, GuildChannel)
    def is_category(self) -> bool: return isinstance(self.current, CategoryChannel)
    def is_forum(self) -> bool: return isinstance(self.current, ForumChannel)
    def is_stage(self) -> bool: return isinstance(self.current, StageChannel)
    def is_voice_channel(self) -> bool: return isinstance(self.current, VoiceChannel)
    def is_text_channel(self) -> bool: return isinstance(self.current, TextChannel)
    def is_thread(self) -> bool: return isinstance(self.current, Thread)
    def is_messageable(self) -> bool: return isinstance(self.current, Messageable)

# Note: run.py doesn't have a way of hooking into its caching mechanism (state.py), just implement it separately
# TODO: Can probably be a lot cleaner - but just to isolate the functionality for now to forward to a db at somepoint
class Cache(Generic[TObject]):

    def __init__(self, entries: Optional[Iterable[TObject]] = None, parent: Optional[Cache] = None):
        self.parent = parent
        self._temp = deque(entries or [])

    @functools.cached_property
    def users(self) -> Cache[User]:
        return Cache()
    @functools.cached_property
    def members(self) -> Cache[Member]:
        return Cache()

    @functools.cached_property
    def reactions(self) -> Cache[Reaction]:
        return Cache()

    @functools.cached_property
    def objects(self) -> Cache[Hashable]:
        return Cache() if self.parent is None else self.parent.objects

    # TODO: Could use channel.type here
    @functools.cached_property
    def messages(self) -> Cache[Message]: return self.objects.filter(lambda o: o.is_message())
    @functools.cached_property
    def guilds(self) -> Cache[Guild]: return self.objects.filter(lambda o: o.is_guild())
    @functools.cached_property
    def channels(self) -> Cache[GuildChannel]: return self.objects.filter(lambda o: o.is_channel())
    @functools.cached_property
    def categories(self) -> Cache[CategoryChannel]: return self.objects.filter(lambda o: o.is_category())
    @functools.cached_property
    def forums(self) -> Cache[ForumChannel]: return self.objects.filter(lambda o: o.is_forum())
    @functools.cached_property
    def stages(self) -> Cache[StageChannel]: return self.objects.filter(lambda o: o.is_stage())
    @functools.cached_property
    def voice_channels(self) -> Cache[VoiceChannel]: return self.objects.filter(lambda o: o.is_voice_channel())
    @functools.cached_property
    def text_channels(self) -> Cache[TextChannel]: return self.objects.filter(lambda o: o.is_text_channel())
    @functools.cached_property
    def threads(self) -> Cache[Thread]: return self.objects.filter(lambda o: o.is_thread())
    @functools.cached_property
    def messageables(self) -> Cache[Messageable]: return self.objects.filter(lambda o: o.is_messageable())

    def count(self) -> int:
        entries = self.entries()
        count = len(entries)
        if count <= 0: return count

        # TODO Could be moved elsewhere
        if entries[0].is_reaction(): # if anything other than reactions are in the current cache, this will not capture that
            return sum(map(lambda entry: entry.current.count, entries))

        return count
    def empty(self) -> bool:
        return self.count() <= 0

    _temp: Deque[CacheEntry[TObject]] # DOESNT WORK WITH MAP/FILTER YET
    def entries(self) -> List[CacheEntry[TObject]]: # todo iterable
        return list(self._temp)
    def current(self) -> Iterable[TObject]: return map(lambda entry: entry.current, self.entries())

    async def push(self, object: TObject) -> None:
        if self.parent: return await self.parent.push(object)

        entry = CacheEntry(current = object)
        cached_entry = get(self.entries(), id=entry.id)
        if cached_entry is not None:
            cached_entry.current = object # TODO; Now it's just last found, this will probably have to be different
            return

        self._temp.append(entry)

    # Helper functions
    def get(self, **attrs: Any) -> Optional[TObject]:
        return get(self.current(), **attrs)
    def get_all(self, **attrs: Any) -> Cache[TObject]:
        return self.filter(lambda entry: get([entry.current], **attrs) is not None)
    def find(self, predicate: Callable[[TObject], bool]) -> Optional[TObject]:
        return find(predicate, self.current())

    # TODO ; These just temps
    def filter(self, predicate: Callable[[CacheEntry[TObject]], bool]) -> Cache[TObject]:
        class FilteredCache(Cache):
            def entries(self) -> List[CacheEntry[TObject]]:
                return list(filter(predicate, self.parent.entries()))

        return FilteredCache(parent=self)
    def map(self, func) -> Cache[TTarget]:
        class MappedCache(Cache):
            def entries(self) -> List[CacheEntry[TObject]]:
                return list(map(func, self.parent.entries()))

        return MappedCache(parent=self)
    def group_by(self, func) -> Cache[TTarget]:
        class GroupByCache(Cache):
            def entries(self) -> List[CacheEntry[TTarget]]:
                # TODO: Values are now an iterable TObject, should be CacheEntry[TObject] use Cache(entries= ) for now
                return [CacheEntry(current=[CacheEntry(current=group), Cache(entries=values)]) for group, values in groupby(self.parent.entries(), func)]
                # return map(lambda grouped_entry: [grouped_entry[0], Cache(entries=grouped_entry[1])], groupby(self.parent.entries(), func))

        return GroupByCache(parent=self)
    def sort(self, func) -> Cache[TTarget]:
        class SortedCache(Cache):
            def entries(self) -> List[CacheEntry[TObject]]:
                return sorted(self.parent.entries(), key=func)

        return SortedCache(parent=self)
    def first(self) -> Optional[CacheEntry[TObject]]: return None if self.empty() else self.entries()[0]
    def last(self) -> Optional[CacheEntry[TObject]]: return None if self.empty() else self.entries()[-1]

def cached(cache: Callable[[Cache], Cache]):
    def decorator(func):
        @functools.wraps(func)
        async def method(self, entry, *args, **kwargs):
            await cache(self.cache).push(entry)
            return await func(self, entry, *args, **kwargs)

        return method

    return decorator

class DiscordTraverser(FunctionQueue):
    cache: Cache

    @dataclass
    class Options:
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,

    def __init__(self, cache: Cache, options: Options = None):
        super().__init__()
        self.cache = cache
        self.options = options

    @cached(lambda cache: cache.reactions)
    async def push_reaction(self, reaction: Reaction):
        # await self.push(reaction.users())
        pass
    @cached(lambda cache: cache.users)
    async def push_user(self, user: User):
        pass
    @cached(lambda cache: cache.members)
    async def push_member(self, member: Member):
        pass

    @cached(lambda cache: cache.messages)
    async def push_message(self, message: Message):
        await self.push(message.reactions)
        # await self.push(message.author)

    @queue
    @cached(lambda cache: cache.messageables)
    async def push_messageable(self, channel: Messageable):
        await self.push(channel.history(
            before=self.options.before,
            after=self.options.after,
            around=None, oldest_first=False, limit=None
        ))

    @queue
    @cached(lambda cache: cache.threads)
    async def push_thread(self, thread: Thread):
        await self.push_messageable(thread)
    @queue
    @cached(lambda cache: cache.channels)
    async def push_channel(self, channel: GuildChannel):
        if isinstance(channel, Messageable): await self.push_messageable(channel)
        # TODO; now happens double for push_guild
        if isinstance(channel, ForumChannel) or isinstance(channel, TextChannel):
            await self.push(channel.threads)

            # Note: guild/channel.threads is only active (last 30ish days), also include older ones
            # TODO: This can probably also be achieved through 'push_message' by checking if it's a thread
            if isinstance(channel, ForumChannel):
                await self.push(channel.archived_threads(limit = None, before = self.options.before))
            if isinstance(channel, TextChannel):
                await self.push(channel.archived_threads(limit = None, before = self.options.before, joined = False, private = False))

    @queue
    @cached(lambda cache: cache.guilds)
    async def push_guild(self, guild: Guild):
        for channel in guild.channels: await self.push_channel(channel)
        # Note: This includes forum threads
        for thread in guild.threads: await self.push_thread(thread)

    @queue
    async def push_iterator(self, iterator: AsyncIterator):
        async for item in iterator: await self.push(item)

    async def push(self, source: Optional[Any]) -> None:
        # print(f'{type(source)}')
        if not source: return

        if isinstance(source, Messageable): await self.push_messageable(source); return
        if isinstance(source, Thread): await self.push_thread(source); return
        if isinstance(source, GuildChannel): await self.push_channel(source); return
        if isinstance(source, Message): await self.push_message(source); return
        if isinstance(source, Guild): await self.push_guild(source); return
        if isinstance(source, Reaction): await self.push_reaction(source); return
        if isinstance(source, User): await self.push_user(source); return
        if isinstance(source, Member): await self.push_member(source); return

        if isinstance(source, AsyncIterator): await self.push_iterator(source); return
        if isinstance(source, Iterable):
            for item in source: await self.push(item)
            return

        raise NotImplementedError(type(source))
