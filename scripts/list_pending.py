#!/usr/bin/env python3
"""Utility: list papers in the tracker that are not done."""
import json

with open('papers/tracker.json') as f:
    t = json.load(f)
papers = t['papers'] if isinstance(t, dict) else t
print('total papers:', len(papers))
for i, p in enumerate(papers):
    s = p.get('status')
    if s != 'done':
        print(i, '|', s, '|', p.get('year'), '|', p.get('title', '')[:80], '|',
              p.get('folder'), '|', p.get('arxiv_id'))
