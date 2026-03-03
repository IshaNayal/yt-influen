"""Test yt-dlp search functionality."""
import json
import yt_dlp
import os

os.environ['PATH'] = r'C:\Users\isha and gaurav\AppData\Roaming\Python\Python314\site-packages\imageio_ffmpeg\binaries;' + os.environ.get('PATH', '')

print('Testing yt-dlp search...')
ydl_opts = {'quiet': False, 'extract_flat': False}
try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        results = ydl.extract_info('ytsearch10:Lu do Magalu', download=False)
        if results and 'entries' in results:
            print(f'\nFound {len(results["entries"])} videos')
            for i, entry in enumerate(results['entries'][:5], 1):
                title = entry.get('title', '?')
                vid_id = entry.get('id', '?')
                print(f'{i}. {title[:70]} (ID: {vid_id})')
        else:
            print('No entries found')
except Exception as e:
    print(f'Error: {e}')
