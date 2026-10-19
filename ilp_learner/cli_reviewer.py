#!/usr/bin/env python3
"""Simple CLI reviewer for GSM8K ILP validation scaffold.
Usage: python3 ilp_learner/cli_reviewer.py [scaffold.jsonl]

Commands during review:
  n  - next example
  p  - previous example
  j <i> - jump to index i (1-based)
  a  - annotate current example's negatives (prompts per candidate)
  s  - save annotations to gsm8k_validation_annotations.json
  q  - quit (prompts to save if unsaved changes)

Annotation input: for each neg candidate choose 1/c/i for correct_negative/incorrect_negative/ambiguous, then optionally add notes.
"""
import sys, json
from pathlib import Path

SCAFFOLD_DEFAULT = Path('ilp_learner/gsm8k_validation_scaffold.jsonl')
OUT_PATH = Path('ilp_learner/gsm8k_validation_annotations.json')

def load_scaffold(path):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    # initialize annotations
    for it in items:
        it.setdefault('_anns', [ {'decision': None, 'notes': ''} for _ in (it.get('neg_candidates') or []) ])
    # try to merge autofill annotations if available
    autofill_path = Path('ilp_learner/gsm8k_validation_annotations_autofill.json')
    if autofill_path.exists():
        try:
            with open(autofill_path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            map_by_id = { str(d.get('id')): d for d in data if d.get('id') is not None }
            for it in items:
                entry = map_by_id.get(str(it.get('id')))
                if not entry:
                    continue
                if 'auto_pred' in entry:
                    it['_auto_pred'] = entry.get('auto_pred')
                if 'annotations' in entry and isinstance(entry['annotations'], list) and len(entry['annotations']) == len(it.get('neg_candidates', [])):
                    it['_anns'] = [ { 'decision': a.get('decision'), 'notes': a.get('notes','') } for a in entry['annotations'] ]
        except Exception:
            pass
    return items

def save_annotations(items, out=OUT_PATH):
    out_list = []
    for it in items:
        out_list.append({'id': it.get('id'), 'neg_candidates': [ { **n, 'annotation': a } for n,a in zip(it.get('neg_candidates', []), it.get('_anns', [])) ]})
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(out_list, fh, indent=2)
    print(f"Saved annotations to {out}")

def pretty_print_example(it, idx, total):
    print(f"\nExample {idx+1}/{total}  id={it.get('id')}  verb={it.get('verb')}  class={it.get('class')}  auto={it.get('_auto_pred','')}")
    print('-'*60)
    print('Background facts:')
    for b in it.get('background', []):
        pred = b.get('pred')
        args = b.get('args', [])
        print(f"  {pred}({','.join(map(str,args))})")
    print('\nPositive head:')
    pos = it.get('pos')
    if pos:
        print(f"  {pos.get('pred')}({','.join(map(str,pos.get('args',[])))})")
    else:
        print('  (none)')
    print('\nNegative candidates:')
    negs = it.get('neg_candidates', [])
    anns = it.get('_anns', [])
    if not negs:
        print('  (none)')
    for i,n in enumerate(negs):
        ann = anns[i] if i < len(anns) else {'decision': None, 'notes': ''}
        print(f"[{i+1}] type={n.get('type')} reason={n.get('reason')}")
        atom = n.get('atom', {})
        print(f"     atom: {atom.get('pred')}({','.join(map(str, atom.get('args',[])))})")
        print(f"     current annotation: {ann.get('decision')} notes: {ann.get('notes')}")


def annotate_example(it):
    negs = it.get('neg_candidates', [])
    if not negs:
        print('No neg candidates to annotate for this example.')
        return
    anns = it.setdefault('_anns', [ {'decision': None, 'notes': ''} for _ in negs ])
    for i,n in enumerate(negs):
        print('\nCandidate #{}/{}:'.format(i+1, len(negs)))
        print('  type:', n.get('type'))
        atom = n.get('atom', {})
        print('  atom:', atom.get('pred'), atom.get('args'))
        cur = anns[i]
        print('  current:', cur)
        while True:
            ch = input('Decision? (1=correct_negative, 2=incorrect_negative, 3=ambiguous, k=keep current, s=skip rest) > ').strip().lower()
            if ch == '1':
                cur['decision'] = 'correct_negative'
                break
            if ch == '2':
                cur['decision'] = 'incorrect_negative'
                break
            if ch == '3':
                cur['decision'] = 'ambiguous'
                break
            if ch == 'k':
                break
            if ch == 's':
                print('Skipping remaining candidates')
                return
            print('Invalid choice')
        note = input('Optional notes (enter to keep): ')
        if note:
            cur['notes'] = note
    print('Finished annotating example')


def main(argv):
    path = Path(argv[1]) if len(argv) > 1 else SCAFFOLD_DEFAULT
    if not path.exists():
        print(f"Scaffold not found at {path}. Ensure you run from repo root or provide path.")
        return
    items = load_scaffold(path)
    total = len(items)
    if total == 0:
        print('No examples found in scaffold')
        return
    idx = 0
    dirty = False
    while True:
        pretty_print_example(items[idx], idx, total)
        cmd = input('\nCommand [n/p/j/a/s/q] > ').strip()
        if not cmd:
            continue
        if cmd == 'n':
            idx = min(total-1, idx+1)
            continue
        if cmd == 'p':
            idx = max(0, idx-1)
            continue
        if cmd.startswith('j'):
            try:
                j = int(cmd.split()[1]) - 1
                if 0 <= j < total:
                    idx = j
                else:
                    print('Out of range')
            except Exception:
                print('Usage: j <index>')
            continue
        if cmd == 'a':
            annotate_example(items[idx])
            dirty = True
            continue
        if cmd == 's':
            save_annotations(items)
            dirty = False
            continue
        if cmd == 'q':
            if dirty:
                ok = input('Unsaved changes. Save before quit? (y/n) > ').strip().lower()
                if ok == 'y':
                    save_annotations(items)
            print('Bye')
            break
        print('Unknown command')

if __name__ == '__main__':
    main(sys.argv)
