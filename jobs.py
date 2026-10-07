import sys, json, os, time, pickle
from crbhmarl.bench import run_episode, METHODS, ATTACKS
S=30
def build():
    J=[]
    E5=[0.05,0.10,0.20,0.30,0.40]
    for m in METHODS:                        # A6-WB supplementary
        for e in E5: J.append(('wb',m,'adaptive_wb',e,30,{},False))
    for f in [4,5,7,8,10,11]:                # breakdown refinement (integer f between grid points)
        for m in METHODS:
            for a in ATTACKS: J.append(('ref',m,a,f/30,30,{},False))
    for m in ['crbhmarl','independent_stores']:   # trust trajectories
        for a in ['colluding','sign_flip','stealthy','adaptive']: J.append(('traj',m,a,0.30,30,{},True))
    for sig in ['fit','grad','temp','nbr','strat']:   # signal ablation
        for a in ATTACKS: J.append(('abl_'+sig,'crbhmarl',a,0.30,30,{'drop_signals':(sig,)},False))
        J.append(('abl_'+sig,'crbhmarl','none',0.0,30,{'drop_signals':(sig,)},False))
    for m in ['median','krum','reputation_only','crbhmarl']:   # non-IID
        for a in ATTACKS: J.append(('noniid',m,a,0.30,30,{'non_iid':True},False))
        J.append(('noniid',m,'none',0.0,30,{'non_iid':True},False))
    for d in [3,10]:                          # topology
        for a in ['colluding','stealthy','fitness_inflation']:
            for e in [0.0]+E5: J.append((f'topo{d}','crbhmarl',a if e>0 else 'none',e,30,{'degree':d},False))
    for key,vals in [('tau_mult',[1,2,4]),('lam',[0.8,0.95,0.99]),('gamma',[2,3])]:   # sensitivity
        for v in vals:
            for a in ['colluding','sign_flip','stealthy','fitness_inflation']: J.append((f'sens_{key}_{v}','crbhmarl',a,0.30,30,{key:v},False))
            J.append((f'sens_{key}_{v}','crbhmarl','none',0.0,30,{key:v},False))
    for key,vals,a in [('q_A5',[0.5,0.95],'stealthy'),('q_A6',[0.5,2.0],'adaptive'),('c',[1.0,5.0],'sign_flip')]:  # severity
        for v in vals:
            for m in ['non_robust','median','reputation_only','crbhmarl']:
                for e in E5: J.append((f'sev_{key}_{v}',m,a,e,30,{key:v},False))
    for N in [10,50,100]:                     # scalability
        for m in ['non_robust','reputation_only','crbhmarl']:
            for a in ['colluding','fitness_inflation']: J.append((f'scale{N}',m,a,0.30,N,{},False))
        for m in ['median','krum']: J.append((f'scale{N}',m,'colluding',0.30,N,{},False))
    return J
J=build()
os.makedirs('jobs',exist_ok=True)
budget=float(sys.argv[1]); t0=time.time(); done=0
for k,(tag,m,a,e,N,cfg,rec) in enumerate(J):
    fn=f'jobs/{k:04d}.json'
    if os.path.exists(fn): continue
    if time.time()-t0>budget: break
    seeds=S if N<100 else 15
    out=[run_episode(m,a,e,N,s,cfg=cfg,record=rec) for s in range(seeds)]
    out=[dict(r,tag=tag) for r in out if r is not None]
    json.dump(out,open(fn,'w')); done+=1
left=sum(1 for k in range(len(J)) if not os.path.exists(f'jobs/{k:04d}.json'))
print('jobs total',len(J),'done now',done,'left',left, round(time.time()-t0))
