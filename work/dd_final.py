"""dd_final: base probe_edgefix_bc_bandv + union of gated dd_pool and dd_stamp diffs (all legit -> phishing_collect)."""
import polars as pl

S = 'subs/'
base = pl.read_csv(S + 'probe_edgefix_bc_bandv.csv')
LABELS = set(base['label'].unique().to_list())
RESERVED = {'AC015935', 'AC279075'}  # rule masks already exclude them; dd_* check_train asserted it


def diff(name):
    s = pl.read_csv(S + name)
    assert s['id'].equals(base['id']), name
    m = s['label'] != base['label']
    return dict(zip(s.filter(m)['id'], s.filter(m)['label']))


acc = [diff('dd_pool.csv'), diff('dd_stamp.csv')]
union = {}
for d in acc:
    for k, v in d.items():
        assert union.setdefault(k, v) == v, k  # overlaps must agree
assert len(union) == 13 and set(union.values()) == {'phishing_collect'}
assert set(base.filter(pl.col('id').is_in(list(union)))['label']) == {'legit'}

out = base.with_columns(pl.col('id').replace_strict(union, default=pl.col('label')).alias('label'))
assert out['id'].equals(base['id']) and out.height == 260582
assert set(out['label'].unique()) <= LABELS
ch = out.filter(out['label'] != base['label'])
assert dict(zip(ch['id'], ch['label'])) == union
assert diff('dd_both.csv') == union  # dd_both is the same union, built independently
out.write_csv(S + 'dd_final.csv')
print('dd_final.csv written, changed', len(union), sorted(union))
