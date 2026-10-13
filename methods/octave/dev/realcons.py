import pandas as pd, numpy as np, json, sys
sys.path.insert(0, '.')
from consist import consistency, MAN
import evalrules as ER
d = pd.read_pickle(sys.argv[1] if len(sys.argv) > 1 else 'rules_real.pkl')
d['id'] = d.key.astype(int)
reps = set(MAN['scenes']['reps'])
base = d[['id', 'ours_h', 'ours_v']].copy()
s0, _ = consistency(base.fillna(1.0), 'ours_h', 'ours_v')
def summ(s):
    return {f"{t}_oct": int(s.loc[t, 'oct']) for t in s.index} | {f"{t}_ag5": round(float(s.loc[t, 'agree5']), 3) for t in s.index}
rows = [dict(variant='frozen', n_changed_reps=0, n_up=0, n_down=0, **summ(s0))]
for nm in ER.VN:
    x = d[['id']].copy()
    x['h'] = d.ours_h * d[nm + '_fx'].fillna(1.0)
    x['v'] = d.ours_v * d[nm + '_fy'].fillna(1.0)
    s, _ = consistency(x.fillna(1.0), 'h', 'v')
    r = d[d.id.isin(reps)]
    fx, fy = r[nm + '_fx'].fillna(1), r[nm + '_fy'].fillna(1)
    rows.append(dict(variant=nm, n_changed_reps=int(((fx != 1) | (fy != 1)).sum()),
                     n_up=int((fx > 1).sum() + (fy > 1).sum()), n_down=int((fx < 1).sum() + (fy < 1).sum()), **summ(s)))
R = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
print(R.to_string())
