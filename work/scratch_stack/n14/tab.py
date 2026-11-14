import sys
for l in open(sys.argv[1], encoding='utf-8', errors='replace'):
    if l.startswith('│') and '---' not in l and ' str ' not in l:
        c = [x.strip() for x in l.strip().strip('│').split('┆')]; print(' | '.join(c[:11] + c[14:15]))
