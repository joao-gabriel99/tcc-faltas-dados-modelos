"""Confere os arquivos extraídos contra o manifesto SHA-256."""
from pathlib import Path
import hashlib
import json
ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'manifesto.json').read_text(encoding='utf-8'))
failures = []
for item in manifest['files']:
    path = ROOT / item['path']
    if not path.is_file():
        failures.append(item['path'] + ': ausente')
        continue
    with path.open('rb') as file:
        digest = hashlib.file_digest(file, 'sha256').hexdigest()
    if digest != item['sha256']:
        failures.append(item['path'] + ': divergente')
if failures:
    raise SystemExit('\n'.join(failures))
print(f"PASS: {len(manifest['files'])} arquivos conferidos.")
