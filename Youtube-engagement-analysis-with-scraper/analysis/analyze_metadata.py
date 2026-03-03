import json
import os

output_dir = r'c:\Users\srini\Desktop\youtube transcripts\output'

# Function to analyze a JSONL file
def analyze_jsonl(filepath):
    metadata = {
        'file_name': os.path.basename(filepath),
        'file_size_mb': round(os.path.getsize(filepath) / (1024 * 1024), 2),
        'total_records': 0,
        'fields': {},
        'sample_record': None
    }
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip():
                record = json.loads(line)
                metadata['total_records'] += 1
                
                # Store first record as sample
                if i == 0:
                    metadata['sample_record'] = record
                
                # Collect field information
                for key, value in record.items():
                    if key not in metadata['fields']:
                        if isinstance(value, (list, dict)):
                            sample = f'{type(value).__name__} ({len(value)} items)'
                        else:
                            sample = value
                        metadata['fields'][key] = {
                            'type': type(value).__name__,
                            'sample_value': sample
                        }
    
    return metadata

# Analyze each file
files = ['videos.jsonl', 'transcripts.jsonl', 'comments.jsonl']
results = {}

for filename in files:
    filepath = os.path.join(output_dir, filename)
    if os.path.exists(filepath):
        print(f'\n{"="*60}')
        print(f'Analyzing {filename}...')
        print('='*60)
        
        results[filename] = analyze_jsonl(filepath)
        meta = results[filename]
        
        print(f'\nFile Size: {meta["file_size_mb"]} MB')
        print(f'Total Records: {meta["total_records"]:,}')
        print(f'\nFields ({len(meta["fields"])}):')
        for field, info in meta['fields'].items():
            print(f'  - {field}: {info["type"]}')
            sample = str(info['sample_value'])
            if len(sample) > 100:
                print(f'      Sample: {sample[:100]}...')
            else:
                print(f'      Sample: {sample}')

print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print(f'Total files: {len(results)}')
total_size = sum(r['file_size_mb'] for r in results.values())
total_records = sum(r['total_records'] for r in results.values())
print(f'Total data size: {total_size:.2f} MB')
print(f'Total records across all files: {total_records:,}')
