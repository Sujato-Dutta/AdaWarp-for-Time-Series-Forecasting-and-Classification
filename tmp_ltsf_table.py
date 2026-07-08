import csv
from pathlib import Path
from collections import defaultdict

root = Path('results')
methods = {
    'AdaWarp-MVPF': [('ltsf_5/MVPF_Cpt/*/metrics/ltsf_main5_combined.csv', 'AdaWarp-MVPF+'), ('ettm2/MVPF/metrics/ltsf_main5_combined.csv', 'AdaWarp-MVPF+')],
    'DLinear': [('ltsf_5/D-Linear/metrics/ltsf_main5_combined.csv', 'DLinear'), ('ettm2/DLinear/metrics/ltsf_main5_combined.csv', 'DLinear')],
    'PatchTST': [('ltsf_5/PatchTST/metrics/ltsf_main5_combined.csv', 'PatchTST'), ('ettm2/PatchTST/metrics/ltsf_main5_combined.csv', 'PatchTST')],
    'TimesNet': [('ltsf_5/TimesNet/metrics/ltsf_main5_combined.csv', 'TimesNet'), ('ettm2/TimesNet/metrics/ltsf_main5_combined.csv', 'TimesNet')],
    'iTransformer': [('ltsf_5/iTransformer/metrics/ltsf_main5_combined.csv', 'iTransformer'), ('ettm2/iTransformer/metrics/ltsf_main5_combined.csv', 'iTransformer')],
    'TimeMixer': [('ltsf_5/TimeMixer/metrics/ltsf_main5_combined.csv', 'TimeMixer'), ('ettm2/Timemixer/metrics/ltsf_main5_combined.csv', 'TimeMixer')],
    'FEDformer': [('ltsf_5/FEDformer/metrics/ltsf_main5_combined.csv', 'FEDformer'), ('ettm2/FEDFormer/metrics/ltsf_main5_combined.csv', 'FEDformer')],
    'VPNet': [('ltsf_5/VPNet/metrics/ltsf_main5_combined.csv', 'VPNet'), ('ettm2/VPNet/metrics/ltsf_main5_combined.csv', 'VPNet')],
}

data = {}
for paper_name, patterns in methods.items():
    for pattern, internal in patterns:
        for path in root.glob(pattern):
            with path.open(newline='') as f:
                for r in csv.DictReader(f):
                    model = r.get('model','')
                    if model != internal:
                        continue
                    key = (r['dataset'], int(r['horizon']))
                    data[(key, paper_name)] = (float(r['mse']), float(r['mae']))

datasets = ['ETTh1','ETTh2','ETTm2','Weather','Electricity','Traffic']
horizons = [96,192,336,720]
print('tasks', len({k for k,m in data}), 'entries', len(data))
for d in datasets:
    for h in horizons:
        missing=[m for m in methods if ((d,h),m) not in data]
        if missing:
            print('MISSING', d,h, missing)

# summary means
print('\nMEANS')
for m in methods:
    vals=[data[((d,h),m)] for d in datasets for h in horizons if ((d,h),m) in data]
    print(m, len(vals), sum(x for x,y in vals)/len(vals), sum(y for x,y in vals)/len(vals))

# latex helpers
order=list(methods.keys())
def fmt_row(d,h):
    vals={m:data[((d,h),m)] for m in order}
    mses=sorted((v[0],m) for m,v in vals.items())
    maes=sorted((v[1],m) for m,v in vals.items())
    best_mse, second_mse = mses[0][1], mses[1][1]
    best_mae, second_mae = maes[0][1], maes[1][1]
    cells=[]
    for m in order:
        mse,mae=vals[m]
        sm=f'{mse:.3f}'
        sa=f'{mae:.3f}'
        if m==best_mse: sm='\\best{'+sm+'}'
        elif m==second_mse: sm='\\second{'+sm+'}'
        if m==best_mae: sa='\\best{'+sa+'}'
        elif m==second_mae: sa='\\second{'+sa+'}'
        cells.append(sm+'/'+sa)
    return f'{d} & {h} & ' + ' & '.join(cells) + r' \\'

print('\nLATEX_ROWS')
for d in datasets:
    for h in horizons:
        print(fmt_row(d,h))

print('\nLATEX_MEAN_ROW')
means={}
for m in order:
    vals=[data[((d,h),m)] for d in datasets for h in horizons]
    means[m]=(sum(x for x,y in vals)/len(vals), sum(y for x,y in vals)/len(vals))
mses=sorted((v[0],m) for m,v in means.items())
maes=sorted((v[1],m) for m,v in means.items())
bm,sm=mses[0][1],mses[1][1]
bma,sma=maes[0][1],maes[1][1]
cells=[]
for m in order:
    mse,mae=means[m]
    a=f'{mse:.3f}'; b=f'{mae:.3f}'
    if m==bm: a='\\best{'+a+'}'
    elif m==sm: a='\\second{'+a+'}'
    if m==bma: b='\\best{'+b+'}'
    elif m==sma: b='\\second{'+b+'}'
    cells.append(a+'/'+b)
print('Mean & -- & ' + ' & '.join(cells) + r' \\')

# win counts by MSE, MAE
print('\nWINS')
for metric_idx,name in [(0,'mse'),(1,'mae')]:
    wins=defaultdict(int)
    seconds=defaultdict(int)
    for d in datasets:
        for h in horizons:
            vals=sorted((data[((d,h),m)][metric_idx],m) for m in order)
            wins[vals[0][1]]+=1
            seconds[vals[1][1]]+=1
    print(name, 'wins', dict(wins), 'seconds', dict(seconds))
