import os  # added by the release export (path configuration)
import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np, json
from synthan import cat
from consist import consistency, MAN
ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
lab=pd.read_csv('synth/labels.csv')[['file','set']]
def synth_tab(dA, dC, nodown=False):
    a=dA.merge(lab,left_on='key',right_on='file'); c=dC.copy(); c['set']='C'
    d=pd.concat([a,c],ignore_index=True)
    out={}
    for st in ('A','B','C'):
        m=d[d.set==st]
        fx=m.octave_fx.where(~(nodown&(m.octave_fx<1)),1.0); fy=m.octave_fy.where(~(nodown&(m.octave_fy<1)),1.0)
        rr=np.r_[m.frozen_h*fx/m.true_h, m.frozen_v*fy/m.true_v]; c0=cat(np.r_[m.frozen_h/m.true_h, m.frozen_v/m.true_v]); c1=cat(rr)
        ch=np.r_[fx!=1, fy!=1]
        out[st]=dict(mape=round(100*np.mean(np.abs(rr-1)),2), w10=round(100*np.mean(c1=='ok'),1), half=int((c1=='half').sum()), dbl=int((c1=='dbl').sum()),
                     fixed=int(((c0!='ok')&(c1=='ok')&ch).sum()), broken=int(((c0=='ok')&(c1!='ok')&ch).sum()), changed=int(ch.sum()))
    return out
def leg(L, nodown=False):
    fx=L.octave_fx.where(~(nodown&(L.octave_fx<1)),1.0); fy=L.octave_fy.where(~(nodown&(L.octave_fy<1)),1.0)
    rr=np.r_[L.frozen_h*fx/L.true_h, L.frozen_v*fy/L.true_v]
    return dict(mape=round(100*np.mean(np.abs(rr-1)),2), w5=round(100*np.mean(np.abs(rr-1)<=0.05),1), w10=round(100*np.mean(np.abs(rr-1)<0.1),1), changed=int((fx!=1).sum()+(fy!=1).sum()))
def real(p, nodown=False):
    p=p.copy()
    if nodown:
        p['octave_fx']=p.octave_fx.where(p.octave_fx>=1,1.0); p['octave_fy']=p.octave_fy.where(p.octave_fy>=1,1.0)
    p['h']=p.frozen_h*p.octave_fx; p['v']=p.frozen_v*p.octave_fy
    s,R=consistency(p.fillna(1.0),'h','v'); s0,R0=consistency(p.fillna(1.0),'frozen_h','frozen_v')
    new=int(((R.oct)&(~R0.oct)).sum()); gone=int(((~R.oct)&(R0.oct)).sum())
    reps=p.id.isin(MAN['scenes']['reps'])
    return dict(**{f'{t}_oct':int(s.loc[t,'oct']) for t in s.index}, **{f'{t}_ag5':round(float(s.loc[t,'agree5']),3) for t in s.index},
                new_oct=new, resolved_oct=gone, reps_up=int(((p.octave_fx>1)&reps).sum()+((p.octave_fy>1)&reps).sum()),
                reps_down=int(((p.octave_fx<1)&reps).sum()+((p.octave_fy<1)&reps).sum()))
V={}
cur=pd.read_csv(ROOT+'/methods/octave/predictions_octave_150.csv')
V['frozen']=dict(synth=synth_tab(pd.read_pickle('mod_synth.pkl').assign(octave_fx=1.0,octave_fy=1.0), pd.read_pickle('mod_synthC.pkl').assign(octave_fx=1.0,octave_fy=1.0)),
                 legacy=leg(pd.read_pickle('mod_legacy.pkl').assign(octave_fx=1.0,octave_fy=1.0)), real=real(cur.assign(octave_fx=1.0,octave_fy=1.0)))
V['ridge_double_only']=dict(synth=synth_tab(pd.read_pickle('mod_synth.pkl'),pd.read_pickle('mod_synthC.pkl'),True), legacy=leg(pd.read_pickle('mod_legacy.pkl'),True), real=real(cur,True))
V['double+halve_posfilter']=dict(synth=synth_tab(pd.read_pickle('mod_synth.pkl'),pd.read_pickle('mod_synthC.pkl')), legacy=leg(pd.read_pickle('mod_legacy.pkl')), real=real(cur))
for m in ('r2h','gsub'):
    V[f'double+halve_{m}']=dict(synth=synth_tab(pd.read_pickle(f'modh_synth_{m}.pkl'),pd.read_pickle(f'modh_synthC_{m}.pkl')), legacy=leg(pd.read_pickle(f'modh_legacy_{m}.pkl')), real=real(pd.read_csv(f'real_h_{m}.csv')))
print(json.dumps(V,indent=0)); json.dump(V,open('variants.json','w'),indent=1)
