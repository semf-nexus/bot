# [SEMF](https://semf.org.es/) - (Discord) Bot

TODO:
- [ ] github action deploy
- [ ] missed ones while bot downtime, recount after x
- [ ] Offline database sync through `cache.py` + events
- [ ] cleanup
- [ ] Move all env variables to global config
- [ ] ensure exclusion of private stuff from counts

### Some useful links
- [Discord Documentation](https://discord.com/developers/docs/)
  - [Discord: Rate Limits](https://discord.com/developers/docs/topics/rate-limits#:~:text=global%22%3A%20true%20%7D-,Global%20Rate%20Limit,rate%20limit%20on%20a%20route.)
- [discord.py](https://github.com/Rapptz/discord.py)

### Local setup

```shell
git clone git@github.com:semf-nexus/bot.git \
&& cd ./bot
```

*Install dependencies*
```shell
python3 -m pip install -U discord.py
```

*Run Discord bot:*
```shell
# DISCORD_SKIP_HOOK=1 Skips manually syncing the Discord Interaction (i.e. AppCommands)`
DISCORD_SKIP_HOOK=0 \
DISCORD_GUILD_ID=844566471501414463 \
BOT_CACHE_GIT_REPOSITORY="git@github.com:semf-nexus/discord-mirror.git" \
BOT_CACHE_GIT_DIRECTORY="./.bot/cache/git" \
BOT_CACHE_GIT_BRANCH="main" \
DISCORD_CLIENT_ID="..." \
DISCORD_TOKEN="..." \
python3 ./bot/run.py
```

*Generating an oauth url to add the bot to a server*
```shell
DISCORD_CLIENT_ID="..." \
DISCORD_TOKEN="..." \
python3 ./bot/oauth2_url.py
```