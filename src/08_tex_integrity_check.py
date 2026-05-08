import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_TEX = os.path.join(ROOT, 'IAQF_column_Final.tex')

INCLUDE_PATTERN = re.compile(r'\\(?:input|include)\{([^}]+)\}')
GRAPHICS_PATTERN = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}')
LABEL_PATTERN = re.compile(r'\\label\{([^}]+)\}')
REF_PATTERN = re.compile(r'\\(?:ref|eqref|autoref|cref|Cref)\{([^}]+)\}')

# Labels with generated artifacts we expect to exist alongside the paper.
TABLE_ARTIFACT_MAP = {
    'tab:arb': ['tables/arbitrage_summary.csv', 'tables/arbitrage_compact.tex'],
    'tab:contagion': ['tables/contagion_intensity.csv', 'tables/contagion_intensity.tex'],
    'tab:regression_hac': ['tables/regression_hac.tex', 'tables/regression_results.txt'],
    'tab:dispersion_vs_adjusted': [
        'tables/dispersion_adjusted_stats.csv',
        'tables/dispersion_adjusted_stats.tex',
    ],
    'tab:half_life_robustness': [
        'tables/half_life_robustness.csv',
    ],
    'tab:coint_vecm': [
        'tables/cointegration_johansen.csv',
        'tables/price_discovery_metrics.csv',
        'tables/cointegration_vecm_merged.tex',
    ],
    'tab:data_coverage': [
        'tables/data_coverage_core.csv',
        'tables/data_coverage_core.tex',
    ],
    'tab:depth_proxy': [
        'tables/depth_proxy_table.csv',
        'tables/depth_proxy_table.tex',
    ],
    'tab:liquidity_spread': [
        'tables/liquidity_spread_table.tex',
    ],
    'tab:ff_sensitivity': [
        'tables/ff_sensitivity_core.csv',
    ],
    'tab:hac_headline': [
        'tables/hac_headline_metrics.csv',
    ],
    'tab:dist_robust': [
        'tables/distributional_robustness.csv',
        'tables/distributional_robustness.tex',
    ],
    'tab:genius_cf': [
        'tables/genius_counterfactual.csv',
        'tables/genius_counterfactual.tex',
    ],
}


def strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        cleaned = []
        for i, ch in enumerate(line):
            if ch == '%' and (i == 0 or line[i - 1] != '\\'):
                break
            cleaned.append(ch)
        out.append(''.join(cleaned))
    return '\n'.join(out)


def resolve_tex_path(path_token: str, base_dir: str) -> str:
    rel = path_token.strip()
    if not rel:
        return ''
    if not rel.endswith('.tex'):
        rel = f'{rel}.tex'
    return os.path.normpath(os.path.join(base_dir, rel))


def resolve_graphics_path(path_token: str, base_dir: str) -> str:
    rel = path_token.strip()
    if not rel:
        return ''
    candidate = os.path.normpath(os.path.join(base_dir, rel))
    if os.path.exists(candidate):
        return candidate
    stem, ext = os.path.splitext(candidate)
    if ext:
        return candidate
    for suffix in ('.png', '.pdf', '.jpg', '.jpeg', '.eps'):
        c = f'{stem}{suffix}'
        if os.path.exists(c):
            return c
    return candidate


def collect_tex_graph(root_tex: str):
    visited = set()
    stack = [root_tex]
    edges = []

    while stack:
        path = stack.pop()
        if path in visited:
            continue
        visited.add(path)
        if not os.path.exists(path):
            continue

        with open(path, 'r', encoding='utf-8') as f:
            text = strip_comments(f.read())
        base_dir = os.path.dirname(path)

        for token in INCLUDE_PATTERN.findall(text):
            resolved = resolve_tex_path(token, base_dir)
            edges.append((path, token, resolved))
            if resolved not in visited:
                stack.append(resolved)

    return visited, edges


def parse_tokens(tex_files):
    labels = []
    refs = []
    graphics_refs = []

    for tex_path in tex_files:
        if not os.path.exists(tex_path):
            continue
        with open(tex_path, 'r', encoding='utf-8') as f:
            text = strip_comments(f.read())
        base_dir = os.path.dirname(tex_path)

        labels.extend(LABEL_PATTERN.findall(text))

        for raw_ref_group in REF_PATTERN.findall(text):
            parts = [p.strip() for p in raw_ref_group.split(',') if p.strip()]
            refs.extend(parts)

        for token in GRAPHICS_PATTERN.findall(text):
            resolved = resolve_graphics_path(token, base_dir)
            graphics_refs.append((tex_path, token, resolved))

    return labels, refs, graphics_refs


def rel(path: str) -> str:
    return os.path.relpath(path, ROOT)


def main():
    errors = []
    warnings = []

    if not os.path.exists(MAIN_TEX):
        print(f'ERROR: missing main TeX file: {MAIN_TEX}')
        sys.exit(1)

    tex_files, include_edges = collect_tex_graph(MAIN_TEX)

    for src, token, resolved in include_edges:
        if not os.path.exists(resolved):
            errors.append(
                f"Missing included TeX file '{token}' referenced in {rel(src)} -> expected {rel(resolved)}"
            )

    labels, refs, graphics_refs = parse_tokens(tex_files)

    for src, token, resolved in graphics_refs:
        if not os.path.exists(resolved):
            errors.append(
                f"Missing figure asset '{token}' referenced in {rel(src)} -> expected {rel(resolved)}"
            )

    referenced_graphics = {os.path.normpath(path) for _, _, path in graphics_refs}
    figures_col_dir = os.path.join(ROOT, 'figures_col')
    if os.path.isdir(figures_col_dir):
        actual_figures = {
            os.path.normpath(os.path.join(figures_col_dir, fname))
            for fname in os.listdir(figures_col_dir)
            if fname.lower().endswith(('.png', '.pdf', '.jpg', '.jpeg', '.eps'))
        }
        unused_figures = sorted(actual_figures - referenced_graphics)
        for fig in unused_figures:
            errors.append(f"Unused figure asset in figures_col not referenced by final TeX: {rel(fig)}")

    label_counts = Counter(labels)
    duplicate_labels = [lbl for lbl, c in label_counts.items() if c > 1]
    for lbl in duplicate_labels:
        errors.append(f"Duplicate label '{lbl}' found {label_counts[lbl]} times")

    label_set = set(labels)
    missing_refs = sorted(set(refs) - label_set)
    for ref in missing_refs:
        errors.append(f"Missing label for reference '{ref}'")

    # If key generated table labels are referenced, ensure backing table artifacts exist.
    ref_and_label_set = set(refs) | label_set
    for label, rel_paths in TABLE_ARTIFACT_MAP.items():
        if label not in ref_and_label_set:
            continue
        for rp in rel_paths:
            full = os.path.join(ROOT, rp)
            if not os.path.exists(full):
                errors.append(f"Missing table artifact for {label}: {rp}")

    allowed_root_tex = {'IAQF_column_Final.tex'}
    allowed_root_pdf = {'IAQF_column_Final.pdf'}
    for fname in os.listdir(ROOT):
        full = os.path.join(ROOT, fname)
        if not os.path.isfile(full):
            continue
        if fname.endswith('.tex') and fname not in allowed_root_tex:
            errors.append(f"Unexpected root-level TeX file: {fname}")
        if fname.endswith('.pdf') and fname not in allowed_root_pdf:
            errors.append(f"Unexpected root-level PDF file: {fname}")
        if fname.endswith(('.aux', '.log', '.out', '.toc', '.fls', '.fdb_latexmk', '.synctex.gz')):
            errors.append(f"Unexpected root-level LaTeX build artifact: {fname}")
        if fname.endswith('.ipynb'):
            errors.append(f"Unexpected root-level notebook artifact: {fname}")

    # Emit compact summary
    print('TeX integrity summary:')
    print(f"  TeX files discovered: {len(tex_files)}")
    print(f"  includegraphics refs: {len(graphics_refs)}")
    print(f"  labels: {len(labels)}")
    print(f"  refs: {len(refs)}")

    if warnings:
        print('\nWarnings:')
        for w in warnings:
            print(f'  - {w}')

    if errors:
        print('\nErrors:')
        for e in errors:
            print(f'  - {e}')
        sys.exit(1)

    print('\nTeX integrity check passed.')


if __name__ == '__main__':
    main()
