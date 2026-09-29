# --------------------------------------------------------
# Transcription of the YieldSAT paper's appendix benchmark tables (PC-07).
#
# Pathak et al., "YieldSAT", CVPR 2026, arXiv:2604.00940, Appendix A.3,
# Tables 13-18: CV10 / LOYO / LORO x field / pixel level, per country-crop
# pair, mean±std across folds. The PDF text layer concatenates each cell as
# "<mean>±<std><next mean>…"; ``split_row`` separates them. Table headers
# ("Evaluation … Level" and the protocol line), not captions, define the
# level and protocol (the Table 16 caption says LORO/field, its header and
# content are LOYO/pixel).
#
#   python -m util.yieldsat_paper_tables paper.txt results/yieldsat/paper_benchmark.csv \
#       [yieldsat.github.io/result/index.html]
#
# The optional project results page (https://yieldsat.github.io/result/,
# means only) is cross-checked: its values are added as web_* columns and
# rows where the two sources differ are flagged (``sources_agree``). In the
# 2026-09 versions 517/540 means agree; the LORO S2+ADM input-fusion
# 3D-ConvLSTM rows of the PDF duplicate the S2 row (Table 17/18 copy error).
#
# ``paper.txt`` is the PDF text layer (e.g. pypdf ``page.extract_text()``,
# pages joined in order). The committed CSV is the output of this parser.
# --------------------------------------------------------

import csv
import re
import sys

PAIRS = ('ARG-C', 'ARG-S', 'ARG-W', 'BRA-C', 'BRA-S', 'BRA-W', 'GER-R', 'GER-W', 'URG-S')
MODELS = ('3D-ConvLSTM', '3D-LSTM', 'LSTM', 'Transformer', 'AFF', 'MMGF')
_NUM = re.compile(r'^-?\d+\.\d+$|^-?\d+$')


def split_row(values):
    """'0.84±0.071.13±0.22…' -> [(0.84, 0.07), (1.13, 0.22), …]."""
    chunks = values.replace(' ', '').split('±')
    means = [chunks[0]]
    stds = []
    for chunk in chunks[1:-1]:
        for decimals in (2, 1):                  # std has 2 decimals unless impossible
            m = re.match(r'^(\d+\.\d{%d})(.*)$' % decimals, chunk)
            if m and _NUM.match(m.group(2)):
                stds.append(m.group(1))
                means.append(m.group(2))
                break
        else:
            raise ValueError('cannot split cell chunk {!r}'.format(chunk))
    stds.append(chunks[-1])
    return [(float(a), float(b)) for a, b in zip(means, stds)]


def parse_tables(text):
    rows = []
    level = protocol = modalities = fusion = None
    table = None
    for raw in text.splitlines():
        line = raw.strip()
        m = re.match(r'^Table (\d+)\.', line)
        if m:
            table = int(m.group(1))
            continue
        if table is None or table < 13 or table > 18:
            continue
        if line.startswith('Evaluation Field-Level'):
            level = 'field'
            continue
        if line.startswith('Evaluation Subfield'):
            level = 'pixel'
            continue
        if line in ('CV10', 'LOYO', 'LORO'):
            protocol = line
            continue
        if line == 'Sentinel-2':
            modalities, fusion = 'S2', 'none'
            continue
        if line == 'Sentinel-2 + ADM':
            modalities = 'S2+ADM'
            continue
        if line in ('Feature Fusion', 'Input Fusion'):
            fusion = line.lower().replace(' ', '_')
            continue
        m = re.match(r'^(X |Feature Fusion |Input Fusion )?(%s) (-?\d.*)$'
                     % '|'.join(re.escape(x) for x in MODELS), line)
        if not m:
            continue
        if m.group(1) and m.group(1).strip() != 'X':
            fusion = m.group(1).strip().lower().replace(' ', '_')
        cells = split_row(m.group(3))
        if len(cells) != 2 * len(PAIRS):
            raise ValueError('table {}: expected 18 cells, got {} in {!r}'.format(table, len(cells), line))
        for i, pair in enumerate(PAIRS):
            (r2, r2s), (rmse, rmses) = cells[2 * i], cells[2 * i + 1]
            rows.append({'source': 'arXiv:2604.00940 Table {}'.format(table), 'protocol': protocol,
                         'level': level, 'modalities': modalities,
                         'fusion': fusion if modalities == 'S2+ADM' else 'none',
                         'model': m.group(2), 'pair': pair, 'r2_mean': r2, 'r2_std': r2s,
                         'rmse_mean': rmse, 'rmse_std': rmses})
    return rows


def parse_results_page(page):
    """{(protocol, level, modalities, fusion, model, pair): (r2, rmse)} from
    the project's results page HTML (36 numbers per model row)."""
    import html as html_mod
    out = {}
    protocol = modalities = fusion = None
    for row in re.findall(r'<tr[^>]*>(.*?)</tr>', page, flags=re.S):
        cells = [html_mod.unescape(re.sub(r'<[^>]+>', '', c)).strip()
                 for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, flags=re.S)]
        nums = [c for c in cells if _NUM.match(c)]
        text = [c for c in cells if not _NUM.match(c)]
        if len(nums) != 36:
            continue
        for t in text:
            if t in ('CV10', 'LORO', 'LOYO'):
                protocol = t
            elif t == 'Sentinel-2':
                modalities, fusion = 'S2', 'none'
            elif t == 'Sentinel-2 + ADM':
                modalities = 'S2+ADM'
            elif t in ('Input Fusion', 'Feature Fusion'):
                fusion = t.lower().replace(' ', '_')
        model = re.sub(r'\s*\[\d+\]', '', text[-1])
        v = [float(x) for x in nums]
        for li, level in enumerate(('field', 'pixel')):
            for i, pair in enumerate(PAIRS):
                out[(protocol, level, modalities, fusion, model, pair)] = (
                    v[li * 18 + 2 * i], v[li * 18 + 2 * i + 1])
    return out


def merge_web(rows, page):
    web = parse_results_page(page)
    for r in rows:
        key = (r['protocol'], r['level'], r['modalities'], r['fusion'], r['model'], r['pair'])
        w = web.get(key)
        r['web_r2_mean'], r['web_rmse_mean'] = w if w else ('', '')
        r['sources_agree'] = (w is not None and abs(w[0] - r['r2_mean']) < 1e-9
                              and abs(w[1] - r['rmse_mean']) < 1e-9)
    return rows


def write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    rows = parse_tables(open(sys.argv[1]).read())
    if len(sys.argv) > 3:
        rows = merge_web(rows, open(sys.argv[3]).read())
        print(sum(not r['sources_agree'] for r in rows), 'rows differ from the results page')
    write_csv(rows, sys.argv[2])
    print(len(rows), 'rows')
