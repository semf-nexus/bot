import os
import datetime
import logging.handlers
from typing import Union, Optional, Sequence

import discord

# https://discordpy.readthedocs.io/en/latest/logging.html
discord.utils.setup_logging(level=logging.INFO)

# https://discordpy.readthedocs.io/en/latest/intents.html
# intents = discord.Intents.default()
intents = discord.Intents.all()
# intents.message_content = True
# intents.members = True
# intents.reactions = True
# intents.guilds = True
# intents.messages = True
# intents.emojis = True
# intents.emojis_and_stickers = True
# intents.moderation = True
# intents.invites = True
# intents.integrations = True
# intents.webhooks = True
# intents.guild_scheduled_events = True

class Client(discord.Client):
    # Client
    async def on_ready(self):
        print(f'We have logged in as {self.user}')

    # Messages
    async def on_message(self, message: discord.Message):
        # if message.author == self.user:
        #     return
        #
        # if message.content.startswith('$hello'):
        #     await message.channel.send('Hello!')
        pass
    async def on_raw_message_edit(self, message: discord.RawMessageUpdateEvent):
        pass
    async def on_raw_message_delete(self, message: discord.RawMessageDeleteEvent):
        pass
    async def on_raw_bulk_message_delete(self, message: discord.RawBulkMessageDeleteEvent):
        pass

    # Reactions
    async def on_raw_reaction_add(self, reaction: discord.RawReactionActionEvent):
        pass
    async def on_raw_reaction_remove(self, reaction: discord.RawReactionActionEvent):
        pass
    async def on_raw_reaction_clear(self, reaction: discord.RawReactionClearEvent):
        pass
    async def on_raw_reaction_clear_emoji(self, reaction: discord.RawReactionClearEmojiEvent):
        pass

    # Members
    async def on_member_join(self, member: discord.Member):
        pass
    async def on_raw_member_remove(self, member: discord.RawMemberRemoveEvent):
        pass
    async def on_member_update(self, before: discord.Member, after: discord.Member):
        pass
    async def on_user_update(self, before: discord.User, after: discord.User):
        pass
    async def on_member_ban(self, guild: discord.Guild, user: Union[discord.User, discord.Member]):
        pass
    async def on_member_unban(self, guild: discord.Guild, user: discord.User):
        pass
    # async def on_presence_update(self, before: discord.Member, after: discord.Member):
    #     pass

    # Channels
    async def on_guild_channel_delete(self, channel: discord.abc.GuildChannel):
        pass
    async def on_guild_channel_create(self, channel: discord.abc.GuildChannel):
        pass
    async def on_guild_channel_pins_update(self, channel: Union[discord.abc.GuildChannel, discord.Thread], last_pin: Optional[datetime.datetime]):
        pass

    # Threads
    async def on_thread_create(self, thread: discord.Thread):
        pass
    async def on_thread_join(self, thread: discord.Thread):
        pass
    async def on_raw_thread_update(self, payload: discord.RawThreadUpdateEvent):
        pass
    async def on_raw_thread_delete(self, payload: discord.RawThreadDeleteEvent):
        pass
    async def on_thread_member_join(self, member: discord.ThreadMember):
        pass
    async def on_thread_member_remove(self, member: discord.ThreadMember):
        pass
    async def on_raw_thread_member_remove(self, payload: discord.RawThreadMembersUpdate):
        pass

    # Integrations (https://support.discord.com/hc/en-us/articles/360045093012-Server-Integrations-Page)
    async def on_integration_create(self, integration: discord.Integration):
        pass
    async def on_integration_update(self, integration: discord.Integration):
        pass
    async def on_guild_integration_update(self, integration: discord.Integration):
        pass
    async def on_guild_integrations_update(self, guild: discord.Guild):
        pass
    async def on_webhooks_update(self, channel: discord.abc.GuildChannel):
        pass
    async def on_raw_integration_delete(self, payload: discord.RawIntegrationDeleteEvent):
        pass

    # Scheduled Events
    async def on_scheduled_event_create(self, event: discord.ScheduledEvent):
        pass
    async def on_scheduled_event_delete(self, event: discord.ScheduledEvent):
        pass
    async def on_scheduled_event_update(self, before: discord.ScheduledEvent, after: discord.ScheduledEvent):
        pass
    async def on_scheduled_event_user_add(self, event: discord.ScheduledEvent, user: discord.User):
        pass
    async def on_scheduled_event_user_remove(self, event: discord.ScheduledEvent, user: discord.User):
        pass

    # Stages
    async def on_stage_instance_create(self, stage_instance: discord.StageInstance):
        pass
    async def on_stage_instance_delete(self, stage_instance: discord.StageInstance):
        pass
    async def on_stage_instance_update(self, stage_instance: discord.StageInstance):
        pass

    # Roles
    async def on_guild_role_create(self, role: discord.Role):
        pass
    async def on_guild_role_delete(self, role: discord.Role):
        pass
    async def on_guild_role_update(self, before: discord.Role, after: discord.Role):
        pass

    # Guilds
    async def on_guild_available(self, guild: discord.Guild):
        pass
    async def on_guild_unavailable(self, guild: discord.Guild):
        pass
    async def on_guild_join(self, guild: discord.Guild):
        pass
    async def on_guild_remove(self, guild: discord.Guild):
        pass
    async def on_guild_update(self, guild: discord.Guild):
        pass
    async def on_guild_emojis_update(self, guild: discord.Guild, before: Sequence[discord.Emoji], after: Sequence[discord.Emoji]):
        pass
    async def on_guild_stickers_update(self, guild: discord.Guild, before: Sequence[discord.GuildSticker], after: Sequence[discord.GuildSticker]):
        pass
    async def on_audit_log_entry_create(self, entry: discord.AuditLogEntry):
        pass
    async def on_invite_create(self, invite: discord.Invite):
        pass
    async def on_invite_delete(self, invite: discord.Invite):
        pass



client = Client(intents=intents)

client.run(os.environ["DISCORD_TOKEN"])