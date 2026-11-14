"""Pick the TabM checkpoint of one seed: best August score after the decoder (aug_dec.py on D). usage: pick_ep.py SEED -> prints the epoch"""
import re, sys
best = max((float(m.group(2)), -int(m.group(1))) for m in (re.search(rf"xTck{sys.argv[1]}e(\d+)\s.*after decoder ([\d.]+)", l) for l in open(f"scratch_mix/tabm_aug{sys.argv[1]}.log")) if m)
print(-best[1])
