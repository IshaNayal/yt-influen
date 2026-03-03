# Utility Scripts

This directory contains helper scripts for channel discovery, data retrospection, and debugging.

## Scripts

### find_channel_ids.py

**Search for YouTube channel IDs by name**

Queries the YouTube Data API to find channel IDs for specific channel names.

**Usage:**

```bash
python utils/find_channel_ids.py
```

The script will prompt you to enter channel names and return their IDs:

```
Enter channel name (or 'quit' to exit): Hautemess Tom
Found channel: Hautemess Tom (UCuAXFkgsw1L7xaCfnd5JJOw)
```

**Common use cases:**

- Initial channel discovery for data collection
- Verifying channel IDs before scraping
- Finding backup channels when primary ones are blocked

### add_channel_ids.py

**Retrospectively add channel_id field to videos.jsonl**

When videos were originally scraped without channel_id, this script adds that field by querying the API.

**Process:**

1. Reads existing `videos.jsonl` file
2. For each configured channel:
   - Queries API for all videos in channel
   - Builds video_id → channel_id mapping
3. Updates each video entry with channel_id
4. Saves to `videos_with_channels.jsonl`

**Usage:**

```bash
python utils/add_channel_ids.py
```

**Configuration:**
Edit the script to include your channel IDs:

```python
channel_ids = [
    'UCuAXFkgsw1L7xaCfnd5JJOw',
    'UC_x5XG1OV2P6uZZ5FSM9Ttw',
]
```

**Why this matters:**
Channel stratification is critical for analysis. Without channel_id, large channels would dominate the dataset and obscure language effects.

### test_captions.py

**Quick IP block detection using known videos**

Tests whether YouTube is blocking your IP by attempting to fetch captions from well-known videos.

**Test videos:**

- "Me at the zoo" (first YouTube video)
- "Never Gonna Give You Up" (Rick Astley)
- "Gangnam Style" (PSY)

**Usage:**

```bash
python utils/test_captions.py
```

**Output:**

```
Testing captions for Me at the zoo (jNQXAC9IVRw)...
✓ Success! First 100 chars: Welcome to the San Diego Zoo's elephant odyssey. Here we have an elephant...

Testing captions for Never Gonna Give You Up (dQw4w9WgXcQ)...
✗ BLOCKED: IpBlocked: YouTube's IP blocking detected
```

**When to use:**

- Before starting a large scraping job
- After encountering transcript extraction errors
- To verify proxy configuration is working

**IP blocking symptoms:**

- `IpBlocked` exceptions
- HTTP 403 errors
- Transcript API timeouts

**Solutions:**

1. Wait 24-48 hours for block to expire
2. Connect to Warp VPN or another proxy
3. Use residential rotating proxies (see main README)

## Common Workflows

### Starting a new scraping project

1. **Find channel IDs:**

   ```bash
   python utils/find_channel_ids.py
   ```

2. **Test for IP blocks:**

   ```bash
   python utils/test_captions.py
   ```

3. **Configure channels in data_collection/main.py**

4. **Start scraping:**
   ```bash
   python data_collection/main.py
   ```

### Fixing missing channel IDs

If you scraped videos before channel stratification was implemented:

1. **Backup original file:**

   ```bash
   cp output/videos.jsonl output/videos_backup.jsonl
   ```

2. **Add channel IDs:**

   ```bash
   python utils/add_channel_ids.py
   ```

3. **Replace original:**
   ```bash
   mv output/videos_with_channels.jsonl output/videos.jsonl
   ```

### Debugging transcript extraction

1. **Check IP status:**

   ```bash
   python utils/test_captions.py
   ```

2. **If blocked, connect VPN and retest**

3. **Verify proxy configuration in data_collection/main.py**

## API Quota Management

All utility scripts use the YouTube Data API, which has daily quotas:

- **Default quota:** 10,000 units per day
- **Search:** 100 units per call
- **Video metadata:** 1 unit per call
- **Channel info:** 1 unit per call

**Tips:**

- Use `find_channel_ids.py` sparingly (100 units per search)
- Batch video metadata requests (50 videos = 1 unit)
- Cache channel ID mappings to avoid repeated lookups
