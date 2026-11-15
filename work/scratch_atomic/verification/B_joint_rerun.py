"""Re-run of agent B's joint test (relationships/s7_joint.py) with the csv written into scratch_atomic/verification instead of
B's directory. Source is B's s6_class_degree.py with B's own two replacements; only the output path differs.
usage (cwd=work, PYTHONPATH=.): python scratch_atomic/verification/B_joint_rerun.py > scratch_atomic/verification/log/B_joint_rerun.log"""
import pathlib
src = pathlib.Path("scratch_atomic/relationships/s6_class_degree.py").read_text(encoding="utf-8")
src = src.replace('legit = y == 0; out = []', 'CELLS = {"ring/burst x n7": CELLS["+sess_n,snd_c1800,ring__pair_n,ring__s4_np_int"] * 10 + np.minimum(n7, 8)}\nM.pop("B4_co_rcv_edge")\nlegit = y == 0; out = []')
assert 'ring/burst x n7' in src and 'O.write_csv(D + "class_degree.csv")' in src
src = src.replace('O.write_csv(D + "class_degree.csv")', 'O.write_csv("scratch_atomic/verification/B_joint_rerun.csv")')
assert "write_" not in src.replace('O.write_csv("scratch_atomic/verification/B_joint_rerun.csv")', "")  # no other file is written
exec(compile(src, "s6_joint_rerun", "exec"))
