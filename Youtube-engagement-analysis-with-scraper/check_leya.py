import json

filepath = r"c:\Users\isha and gaurav\OneDrive\Desktop\yt-influe\Youtube-engagement-analysis-with-scraper\output\ai_influencers\Leya Love\leya-love-transcripts.jsonl"

with open(filepath, 'r', encoding='utf-8') as f:
    lines = [line for line in f if line.strip()]
    print(f'Total transcripts: {len(lines)}\n')
    
    if lines:
        for i in range(min(3, len(lines))):
            parts = lines[i].split(' ', 1)
            if len(parts) > 1 and parts[0].isdigit():
                data = json.loads(parts[1])
                print(f'[#{parts[0]}] Title: {data.get("title", "N/A")[:70]}')
                trans = data.get('transcript', '')
                print(f'    Sample: {trans[:120]}...\n')
