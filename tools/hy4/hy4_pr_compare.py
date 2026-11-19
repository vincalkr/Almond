import numpy as np, sys, os
# usage: hy4_pr_compare.py <dir with al/ (HY4_DUMP_DIR) and pr/ (hy4_pr_dump) subdirs>
S=sys.argv[1]
def ld(d,n): return np.fromfile(os.path.join(S,d,n+'.bin'),dtype='<f4')
names=['hc_init']+[f'l_out-{i}' for i in range(42)]+['result_output']
print(f"{'tensor':14s} {'rel_l2':>9s} {'max_abs':>9s} {'cos':>8s} {'ref_rms':>9s}")
for n in names:
    a,b=ld('al',n),ld('pr',n)
    if a.size!=b.size: print(n,'size',a.size,b.size); continue
    d=a-b; rel=np.linalg.norm(d)/np.linalg.norm(b); cos=a@b/np.linalg.norm(a)/np.linalg.norm(b)
    print(f"{n:14s} {rel:9.2e} {np.abs(d).max():9.3g} {cos:8.5f} {np.sqrt((b*b).mean()):9.3g}")
a=ld('al','result_output').reshape(4,-1); b=ld('pr','result_output').reshape(4,-1)
print('argmax al',a.argmax(1),'pr',b.argmax(1))
for t in range(4):
    ta=set(np.argsort(-a[t])[:10]); tb=set(np.argsort(-b[t])[:10]); print('tok',t,'top10 overlap',len(ta&tb))
