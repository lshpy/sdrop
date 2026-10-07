"""Best-validation worst-group / average for Waterbirds histories. Usage: python summarize_wb.py <dir> [regex]"""
import sys, glob, re, pandas as pd
rows = []
for f in sorted(glob.glob(f'{sys.argv[1]}/waterbirds_*_history.csv')):
    rid = f.split('/')[-1].replace('_history.csv', '')
    if len(sys.argv) > 2 and not re.search(sys.argv[2], rid): continue
    h = pd.read_csv(f); be = int(h['best_epoch'].iloc[-1]); r = h[h['epoch'] == be].iloc[0]
    rows.append({'run': rid, 'seed': int(re.search(r'seed(\d+)', rid).group(1)), 'best_epoch': be,
                 'wg': r['test_worst_group'], 'avg': r['test_acc'], 'avg_group': r['test_avg_group']})
df = pd.DataFrame(rows); df['arm'] = df.run.str.replace(r'_seed\d+', '', regex=True).str.replace('waterbirds_', '')
pd.set_option('display.width', 200); print(df[['arm', 'seed', 'best_epoch', 'wg', 'avg', 'avg_group']].round(2).to_string(index=False))
print(df.groupby('arm')[['wg', 'avg']].agg(['mean', 'std', 'count']).round(2).to_string())
