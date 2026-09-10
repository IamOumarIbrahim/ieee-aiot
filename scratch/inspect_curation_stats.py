import json
import glob
for p in sorted(glob.glob('runs/curation_stats/*.json')):
    print('='*70)
    print(p)
    with open(p) as f:
        print(f.read())
