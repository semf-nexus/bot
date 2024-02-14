from datetime import datetime
from enum import Enum
from typing import Optional, Union

import discord
from dateutil import parser
from discord import PartialEmoji, Emoji, Message
from discord.ext.commands import Converter, Context, BadArgument, EmojiConverter


# TODO: Python doesn't have a built-in thing for this?
class DatetimeConverter(Converter):
    async def convert(self, ctx: Context, argument: str):
        try:
            return parser.parse(argument)
        except:
            raise BadArgument(f'"{argument}" could not be parsed to a date')

# https://discord.com/developers/docs/reference#message-formatting-timestamp-styles
class TimestampStyle(Enum): t = 't'; T = 'T'; d = 'd'; D = 'D'; f = 'f'; F = 'F'; R = 'R'
def discord_timestamp(date: Optional[datetime], style: TimestampStyle = TimestampStyle.f, default = "") -> str:
    if date is None: return default
    return f'<t:{int(date.timestamp())}:{style.value}>'

async def lookup_emoji(ctx: Context, emoji: Union[PartialEmoji, Emoji, str]) -> Emoji:
    if isinstance(emoji, Emoji): return emoji
    return await EmojiConverter().convert(ctx = ctx, argument = str(emoji))

def to_embed(message: Message):
    embed = discord.Embed(
        timestamp=message.created_at,
        url=message.jump_url,
        type='rich',
        title=message.channel.name,
        description=message.content,
    )

    # TODO: There's no better alternative for this? - should open profile in modal?
    url = f'https://discord.com/users/{message.author.id}'
    embed.set_author(
        name=message.author.display_name,
        url=url,
        icon_url=message.author.display_icon.url if message.author.display_icon is not None else None
    )

    return embed