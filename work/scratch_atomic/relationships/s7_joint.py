"""Agent B: joint test for pattern c. Cell = base x ring/burst bins x exact recent 7 d count n7 (cap 8). Does C5 / C8 survive both together?"""
import re, pathlib
src = pathlib.Path("scratch_atomic/relationships/s6_class_degree.py").read_text(encoding="utf-8")
src = src.replace('legit = y == 0; out = []', 'CELLS = {"ring/burst x n7": CELLS["+sess_n,snd_c1800,ring__pair_n,ring__s4_np_int"] * 10 + np.minimum(n7, 8)}\nM.pop("B4_co_rcv_edge")\nlegit = y == 0; out = []')
src = src.replace('class_degree.csv', 'joint.csv')
assert 'ring/burst x n7' in src
exec(compile(src, "s6_joint", "exec"))
