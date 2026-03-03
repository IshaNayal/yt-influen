import pandas as pd
import json
import os

# Helper to convert a JSONL file to Excel in the same folder

def jsonl_to_excel(jsonl_path):
    xlsx_path = os.path.splitext(jsonl_path)[0] + ".xlsx"
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Strip leading line numbers (e.g., "1 {" -> "{")
            if line[0].isdigit():
                idx = line.find('{')
                if idx != -1:
                    line = line[idx:]
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping invalid line in {jsonl_path}:", line[:100])
    if not data:
        print(f"No valid data found in {jsonl_path}.")
        return
    df = pd.DataFrame(data)
    df.to_excel(xlsx_path, index=False)
    print(f"Converted {jsonl_path} to {xlsx_path}")

# Convert all three influencer files
jsonl_to_excel(r"C:/Users/isha and gaurav/OneDrive/Desktop/yt-influe/Youtube-engagement-analysis-with-scraper/output/ai_influencers/Lu do Magalu/lu-do-magalu-transcripts.jsonl")
jsonl_to_excel(r"C:/Users/isha and gaurav/OneDrive/Desktop/yt-influe/Youtube-engagement-analysis-with-scraper/output/ai_influencers/Leya Love/leya-love-transcripts.jsonl")
jsonl_to_excel(r"C:/Users/isha and gaurav/OneDrive/Desktop/yt-influe/Youtube-engagement-analysis-with-scraper/output/ai_influencers/Lil Miquela/lil-miquela-transcripts.jsonl")