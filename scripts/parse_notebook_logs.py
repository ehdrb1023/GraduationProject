#!/usr/bin/env python3
"""노트북 출력 로그에서 에폭별 지표를 뽑아 results/parsed/*.csv 로 저장한다.
리포 루트에서 실행: python3 scripts/parse_notebook_logs.py"""
import json, glob, os, re, csv

OUT = 'results/parsed'
pat_ep   = re.compile(r'Ep(?:och)?\s*\[(\d+)/(\d+)\]')
pat_tag  = re.compile(r'^\[([^\]]+)\]')
fields = {
    'time':       re.compile(r'(?:Time:\s*)?([\d.]+)s'),
    'train_loss': re.compile(r'(?:Train )?Loss:\s*([\d.]+)'),
    'train_acc':  re.compile(r'Train(?: Acc)?:\s*([\d.]+)%'),
    'test_acc':   re.compile(r'(?:Test|Eval)(?: Acc)?:\s*([\d.]+)%'),
    'lr':         re.compile(r'lr:\s*([\d.eE+-]+)'),
    'rho':        re.compile(r'rho:\s*([\d.]+)'),
    'sharpness':  re.compile(r'Sharp(?:ness)?:\s*([\d.-]+)'),
    'surrogate_gap': re.compile(r'Surrogate Gap:\s*([\d.]+)'),
}

index = []
for nb_path in sorted(glob.glob('GraduationProject/**/*.ipynb', recursive=True)):
    nb = json.load(open(nb_path))
    base = os.path.splitext(os.path.basename(nb_path))[0].replace(' ', '_')
    for ci, cell in enumerate(nb['cells']):
        lines = []
        for o in cell.get('outputs', []):
            t = ''.join(o.get('text', []))
            lines += t.splitlines()
        runs, cur, last_ep = [], [], 0
        for ln in lines:
            m = pat_ep.search(ln)
            if not m:
                continue
            ep = int(m.group(1))
            if ep <= last_ep and cur:      # 에폭이 되돌아가면 새 run
                runs.append(cur); cur = []
            last_ep = ep
            row = {'epoch': ep, 'total_epochs': int(m.group(2))}
            tm = pat_tag.match(ln.strip())
            row['tag'] = tm.group(1) if tm else ''
            for k, rx in fields.items():
                mm = rx.search(ln)
                row[k] = mm.group(1) if mm else ''
            cur.append(row)
        if cur:
            runs.append(cur)
        for ri, run in enumerate(runs):
            if len(run) < 3:
                continue
            tags = sorted({r['tag'] for r in run if r['tag']})
            name = f'{base}__c{ci}' + (f'_r{ri}' if len(runs) > 1 else '')
            path = os.path.join(OUT, name + '.csv')
            cols = ['epoch','total_epochs','tag','time','train_loss','train_acc',
                    'test_acc','lr','rho','sharpness','surrogate_gap']
            with open(path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(run)
            finals = [r for r in run if r['test_acc']]
            index.append({
                'file': name + '.csv', 'notebook': nb_path, 'cell': ci,
                'epochs': len(run), 'tags': '|'.join(tags),
                'final_test_acc': finals[-1]['test_acc'] if finals else '',
                'best_test_acc': max((float(r['test_acc']) for r in finals), default=''),
                'final_sharpness': next((r['sharpness'] for r in reversed(run) if r['sharpness']), ''),
            })

with open(os.path.join(OUT, '_index.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(index[0].keys())); w.writeheader(); w.writerows(index)
print(f'{len(index)}개 run 추출')
