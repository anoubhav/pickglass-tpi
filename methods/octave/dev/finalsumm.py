import os  # added by the release export (path configuration)
import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np, json
from synthan import cat
from consist import consistency, MAN, PAIRS
ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
lab=pd.read_csv('synth/labels.csv')[['file','set']]
def met(rr, c0=None, ch=None):
    c=cat(rr); o=dict(n_axes=int(len(rr)), MAPE=round(100*np.mean(np.abs(rr-1)),2), within5=round(100*np.mean(np.abs(rr-1)<=0.05),1),
                      within10=round(100*np.mean(c=='ok'),1), half_axes=int((c=='half').sum()), double_axes=int((c=='dbl').sum()))
    if c0 is not None:
        o.update(changed=int(ch.sum()), fixed=int(((c0!='ok')&(c=='ok')&ch).sum()), broken=int(((c0=='ok')&(c!='ok')&ch).sum()))
    return o
a=pd.read_pickle('mod_synth.pkl').merge(lab,left_on='key',right_on='file'); c=pd.read_pickle('mod_synthC.pkl'); c['set']='C'
d=pd.concat([a,c],ignore_index=True)
out={"synthetic":{}}
names={'A':'A: seeds 990000-990299, default v2 generator','B':'B: seeds 991000-991299, + oxford / oxfordw paired yarns, more basket','C':'C: seeds 992000-992149, unbalanced (ratio 1.6-2.4) plain / twill'}
for st in ('A','B','C','all'):
    m=d if st=='all' else d[d.set==st]
    r0=np.r_[m.frozen_h/m.true_h, m.frozen_v/m.true_v]; r1=np.r_[m.h_count/m.true_h, m.v_count/m.true_v]
    ch=np.r_[m.octave_fx!=1, m.octave_fy!=1]
    out["synthetic"][names.get(st,'all 750 renders')]={'n_images':int(len(m)),'frozen_ours':met(r0),'octave_v1':met(r1,cat(r0),ch)}
L=pd.read_pickle('mod_legacy.pkl')
r0=np.r_[L.frozen_h/L.true_h, L.frozen_v/L.true_v]; r1=np.r_[L.h_count/L.true_h, L.v_count/L.true_v]
out["legacy13"]={'frozen_ours':met(r0),'octave_v1':met(r1,cat(r0),np.r_[L.octave_fx!=1,L.octave_fy!=1]),
                 'note':'this pipeline (legacy.rectify_legacy + frozen estimate) gives 9.23% for frozen ours; the paper figure 11.6% uses the legacy13 protocol of methods/results'}
p=pd.read_csv(ROOT+'/methods/octave/predictions_octave_150.csv')
cons={}
s0,R0=consistency(p.fillna(1.0),'frozen_h_count','frozen_v_count'); s1,R1=consistency(p.fillna(1.0),'h_count','v_count')
for nm,s in (('frozen_ours',s0),('octave_v1',s1)):
    cons[nm]={t:{'n_axes':int(s.loc[t,'n_axes']),'octave_2x_axes':int(s.loc[t,'oct']),'x3_axes':int(s.loc[t,'x3']),'agree5_axes':round(float(s.loc[t,'agree5']),3),
                 'n_pairs':int(s.loc[t,'n_pairs']),'pairs_with_2x':int(s.loc[t,'pairs_oct']),'pairs_both5':round(float(s.loc[t,'pairs_both5']),3)} for t in s.index}
cons['new_2x_disagreements']=int(((R1.oct)&(~R0.oct)).sum()); cons['resolved_2x_disagreements']=int(((~R1.oct)&(R0.oct)).sum())
cons['definition']='pairs: reshoot = 41 accepted same-patch pairs of scene reps (paper/tables/fabric_groups_v2.json; |angle|<15 same axes, 75-90 swapped, else sorted); res = quarter_res_copy vs rep; rot = members with scene_rot_k != 0 vs rep; copy = png/byte/notebook copies vs rep (manifest). Per axis r = max/min of the two counts; 2x if 1.70 <= r <= 2.35.'
out["real150_label_free_consistency"]=cons
reps=p.id.isin(MAN['scenes']['reps'])
out["real150_changes"]={'files_changed_axes':int((p.octave_fx!=1).sum()+(p.octave_fy!=1).sum()),'rep_axes_doubled':int(((p.octave_fx>1)&reps).sum()+((p.octave_fy>1)&reps).sum()),
   'rep_axes_halved':int(((p.octave_fx<1)&reps).sum()+((p.octave_fy<1)&reps).sum()),
   'changed_files':p[(p.octave_fx!=1)|(p.octave_fy!=1)][['id','frozen_h_count','frozen_v_count','h_count','v_count','octave_fx','octave_fy']].round(2).to_dict('records')}
print(json.dumps(out,indent=1)); json.dump(out,open('final_summary.json','w'),indent=1)
