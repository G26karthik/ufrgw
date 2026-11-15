"""Print the per-class precision / recall / F1 blocks of chosen comparisons from the va_score.py report logs, as markdown code blocks.
usage (cwd=work): python scratch_atomic/verification/va_percls.py "N aug van7 vs C [q]" "D sep van7dm vs A_C [q]" ..."""
import glob, sys

L = [ln.rstrip("\n") for f in sorted(glob.glob("scratch_atomic/verification/log/score_*.log")) for ln in open(f, encoding="utf8", errors="replace")]
for key in sys.argv[1:]:
    hit = [i for i, ln in enumerate(L) if ln.startswith(key)]
    if not hit: print(f"(no block for {key})"); continue
    i = hit[-1]; print("```"); print(L[i][:150])
    for ln in L[i + 2:i + 15]: print(ln[2:])   # header + 12 classes (line i+1 holds the counts, shown in the summary table)
    print("```")
