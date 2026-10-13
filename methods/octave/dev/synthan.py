import pandas as pd, numpy as np
def load():
    d=pd.read_pickle('lean_synth.pkl'); lab=pd.read_csv('synth/labels.csv')
    d=d.merge(lab[['file','weave','colour_style','set','r_eff','epi','ppi','blur_sigma_px','glare']],left_on='key',right_on='file')
    d['pxh']=d.S/d.true_h; d['pxv']=d.S/d.true_v
    return d
def cat(r):
    l=np.log2(np.asarray(r,float))
    return np.where(np.abs(l)<np.log2(1.1),'ok',np.where(np.abs(l+1)<0.3,'half',np.where(np.abs(l-1)<0.3,'dbl','other')))
def long(d, hcol='ours_h', vcol='ours_v'):
    a=d[['set','weave','colour_style','pxh']].copy().rename(columns={'pxh':'px'}); a['ax']='h'; a['r']=d[hcol]/d.true_h
    b=d[['set','weave','colour_style','pxv']].copy().rename(columns={'pxv':'px'}); b['ax']='v'; b['r']=d[vcol]/d.true_v
    X=pd.concat([a,b],ignore_index=True); X['c']=cat(X.r); X['ape']=100*np.abs(X.r-1); return X
if __name__=='__main__':
    d=load(); X=long(d)
    print(pd.crosstab(X.set, X.c, margins=True)); print(pd.crosstab([X.set,X.weave], X.c)); print(pd.crosstab(X.colour_style, X.c))
    print(X.groupby('set').ape.agg(['mean','median']))
