"""Split the mistyped compartment (label-hidden Sep benchmark, final decode) into look-alike pairs and the uniform rest
(train labels carry about 4% random wrong fraud types, which the clean test labels do not). cwd=work, PYTHONPATH=."""
import numpy as np, polars as pl
import train as T
z = np.load("scratch_gap/dec_N.npz"); y, q = z["y"], z["C_q"]; K = T.K
C = np.zeros((K, K), int); np.add.at(C, (y, q), 1)
F = C[1:, 1:].copy(); np.fill_diagonal(F, 0)
pairs = sorted(((int(F[i, j]), T.LABELS[i + 1], T.LABELS[j + 1]) for i in range(K - 1) for j in range(K - 1) if F[i, j]), reverse=True)
print("mistyped rows", F.sum(), "| pairs with >= 30 rows:", [p for p in pairs if p[0] >= 30])
small = [p[0] for p in pairs if p[0] < 30]
print(f"other pairs: {len(small)} pairs, {sum(small)} rows, mean {np.mean(small):.1f} rows per pair (110 possible pairs)")
right = np.trace(C[1:, 1:]); print(f"fraud-labelled rows predicted fraud: {C[1:, 1:].sum()}, typed as labelled {right}; 4% random-type noise would give about {0.04 * C[1:, 1:].sum():.0f} mistyped rows")
f1 = lambda p: T.macro_f1(y, p).mean()
big = {(a, b) for n, a, b in pairs if n >= 30}
m_big = np.array([(T.LABELS[a], T.LABELS[b]) in big for a, b in zip(y, q)]) & (y > 0) & (q > 0) & (y != q)
m_all = (y > 0) & (q > 0) & (y != q)
for name, m in (("look-alike pairs (>= 30 rows)", m_big), ("all other mistyped rows", m_all & ~m_big)):
    p = q.copy(); p[m] = y[m]; print(f"oracle fix of {name}: rows {int(m.sum())}, macro {f1(q):.4f} -> {f1(p):.4f} ({f1(p) - f1(q):+.4f})")
