import sys, json, time, os
from crbhmarl.bench import run_episode, ATTACKS
S=30; N=30; ETAS=[0.05,0.10,0.20,0.30,0.40]
kind=sys.argv[1]; methods=sys.argv[2].split(',')
os.makedirs('res',exist_ok=True)
for m in methods:
    t=time.time(); out=[]
    if kind=='main':
        for s in range(S): out.append(run_episode(m,'none',0.0,N,s))
        for a in ATTACKS:
            for e in ETAS:
                for s in range(S): out.append(run_episode(m,a,e,N,s))
    elif kind=='wb':
        for e in ETAS:
            for s in range(S): out.append(run_episode(m,'adaptive_wb',e,N,s))
    out=[r for r in out if r is not None]
    json.dump(out,open(f'res/{kind}_{m}.json','w'))
    print(kind,m,len(out),round(time.time()-t,1),flush=True)
