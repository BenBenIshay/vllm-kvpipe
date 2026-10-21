import json,glob,numpy as np,statistics as st
BT=262144; L=64; X=25.1e9; BWP=1935e9; KP=4.55
def load(p): return json.load(open(p))
def rows(d):
    R=[x for x in d['rows'] if x.get('usable',True)]
    r0=[x for x in R if x['riders']==0][0]['step_ms']; return r0,R
cm=0.497; sl=0.2121
def c_minb(S): return cm+sl*S/1000
def c_real(S,cmin=None):
    cmin=c_minb(S) if cmin is None else cmin
    by=S*BT
    return cmin+KP*by/BWP*1e3+by/(L*X)*1e3, KP*by/BWP*1e3, by/(L*X)*1e3
out=[]
F1=5.0
runs=sorted(glob.glob('/mnt/beegfs/benishay/paper_runs/20261019/m22c/KK_*.json'))+sorted(glob.glob('/mnt/beegfs/benishay/paper_runs/20261020/m23/RB_real*.json'))
for p in runs:
    d=load(p); T0,R=rows(d); cp=c_real(1024)[0]
    for x in R:
        r=x['riders']
        if 1<=r<=4: out.append(('real',p.split('/')[-1],r,T0,T0+F1+cp*r,x['step_ms']))
for p in sorted(glob.glob('/mnt/beegfs/benishay/paper_runs/20261020/m23/RB_minb*.json')):
    d=load(p); T0,R=rows(d)
    for x in R:
        r=x['riders']
        if r>=1: out.append(('minb',p.split('/')[-1],r,T0,T0+F1+c_minb(1024)*r,x['step_ms']))
# no-contention model for comparison (paper before E1)
for k in ('real','minb'):
    ee=np.array([(o[4]-o[5])/o[5]*100 for o in out if o[0]==k]); dd=np.array([(o[4]-o[5]) for o in out if o[0]==k])
    print(k,'n',len(ee),'step err pct mean %.2f max %.2f ; ms mean %.2f max %.2f'%(ee.mean(),abs(ee).max(),dd.mean(),abs(dd).max()))
    # gain error
    ge=np.array([((1+o[2])/o[4]*o[3]-(1+o[2])/o[5]*o[3])/((1+o[2])/o[5]*o[3])*100 for o in out if o[0]==k])
    print('   gain err pct mean %.2f max %.2f'%(ge.mean(),abs(ge).max()))
nc=[(o[3]+F1+c_minb(1024)*o[2]-o[5])/o[5]*100 for o in out if o[0]=='real']
print('real predicted WITHOUT contention+l0: mean %.2f max %.2f'%(np.mean(nc),max(abs(x) for x in nc)))
D='/mnt/beegfs/benishay/paper_runs/20260930/overhead/22_c_context'
def fitc(p,maxload):
    d=load(p);T0,R=rows(d); lr=(5.71*d['ctx']/512)
    pts=[(x['riders'],x['step_ms']-T0) for x in R if x['riders']>0 and x['riders']*lr<=maxload*T0]
    r=np.array([a for a,b in pts]);y=np.array([b for a,b in pts]); c,F=np.polyfit(r,y,1); return c,F,T0
b16=[]
for ctx in (128,256,512):
    mb=[fitc(p,9) for p in sorted(glob.glob(D+'/MB_c%d_*.json'%ctx))]
    cc=[fitc(p,0.75) for p in sorted(glob.glob(D+'/CC_c%d_*.json'%ctx))]
    cmb=st.mean(x[0] for x in mb); pr=c_real(ctx,cmb)
    b16.append((ctx,cmb,pr[0],st.mean(x[0] for x in cc),min(x[0] for x in cc),max(x[0] for x in cc)))
    print('B16 ctx%d: c_minb %.3f pred c_real %.3f (cont %.3f l0 %.3f) meas mean %.3f range %.3f-%.3f; T0 %.2f'%(ctx,cmb,pr[0],pr[1],pr[2],st.mean(x[0] for x in cc),min(x[0] for x in cc),max(x[0] for x in cc),st.mean(x[2] for x in cc)))
W=65.53e9
def t0(B,S): return W/(BWP*0.75)*1e3+B*S*BT/(BWP*0.64)*1e3
print('T0 pred B1 c1024 %.2f B16 c128 %.2f c512 %.2f B1 c14724 %.2f'%(t0(1,1024),t0(16,128),t0(16,512),t0(1,14724)))
json.dump(dict(points=out,b16=b16),open('val_points.json','w'))
