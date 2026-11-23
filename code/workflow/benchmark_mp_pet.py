#!/home/zhzhan/miniconda3/envs/mattersim/bin/python
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from common import PROJECT, normalize_formula
from aggregate_recommendations import aggregate_recommendations

DEFAULT_TARGETS = [
    'KI', 'NaCl', 'KBr', 'CsCl', 'MgO', 'CaO', 'Al2O3', 'SiC', 'TiO2', 'ZnS',
    'GaN', 'BN', 'FeS2', 'Cu2O', 'ZrO2', 'BaTiO3', 'SrTiO3', 'LiFePO4',
    'LiCoO2', 'CaCO3', 'PbTe', 'CdTe', 'NiO', 'Fe2O3', 'LaAlO3'
]

SUBJOURNAL_TARGETS = [
    'LiF', 'NaF', 'KF', 'CsF', 'LiCl', 'NaCl', 'KCl', 'CsCl', 'LiBr', 'NaBr',
    'KBr', 'CsBr', 'LiI', 'NaI', 'KI', 'CsI', 'MgF2', 'CaF2', 'BaF2', 'AlF3',
    'Li2O', 'Na2O', 'K2O', 'MgO', 'CaO', 'SrO', 'BaO', 'Al2O3', 'SiO2', 'TiO2',
    'ZrO2', 'HfO2', 'ZnO', 'Cu2O', 'CuO', 'NiO', 'CoO', 'MnO', 'FeO', 'Fe2O3',
    'Fe3O4', 'Cr2O3', 'V2O5', 'SnO2', 'CeO2',
    'BN', 'AlN', 'GaN', 'InN', 'Si3N4', 'TiN', 'ZrN', 'VN', 'NbN', 'SiC',
    'TiC', 'ZrC', 'WC', 'B4C', 'MgB2', 'TiB2', 'ZnS', 'CdS', 'PbS', 'FeS2',
    'MoS2', 'WS2', 'CdTe', 'PbTe', 'Bi2Te3',
    'BaTiO3', 'SrTiO3', 'CaTiO3', 'PbTiO3', 'KNbO3', 'NaNbO3', 'LaAlO3', 'LaFeO3', 'LaMnO3', 'LaCoO3',
    'YAlO3', 'MgAl2O4', 'ZnAl2O4', 'CoAl2O4', 'MnFe2O4', 'CoFe2O4', 'NiFe2O4', 'BaZrO3', 'SrZrO3', 'LiNbO3',
    'LiTaO3', 'BiFeO3', 'CaMnO3', 'SrMnO3', 'BaSnO3',
    'LiFePO4', 'LiCoO2', 'LiMn2O4', 'LiNiO2', 'Li4Ti5O12', 'NaFePO4', 'NaCoO2', 'NaMnO2', 'CaCO3', 'MgCO3',
    'BaCO3', 'SrCO3', 'Li2CO3', 'Na2CO3', 'K2CO3', 'CaSO4', 'BaSO4', 'SrSO4', 'Li2SO4', 'Na2SO4',
    'K2SO4', 'MgSiO3', 'CaSiO3', 'Li2SiO3', 'Na2SiO3',
]

TARGET_SETS = {
    'core25': DEFAULT_TARGETS,
    'subjournal120': SUBJOURNAL_TARGETS,
}

CATEGORY_GROUPS = {
    'Halides': SUBJOURNAL_TARGETS[0:20],
    'Binary oxides': SUBJOURNAL_TARGETS[20:45],
    'Nitrides/carbides/chalcogenides': SUBJOURNAL_TARGETS[45:70],
    'Ternary oxides': SUBJOURNAL_TARGETS[70:95],
    'Battery/salts/silicates': SUBJOURNAL_TARGETS[95:120],
}

CATEGORY_LOOKUP = {
    normalize_formula(target): category
    for category, targets in CATEGORY_GROUPS.items()
    for target in targets
}

METRIC_COLUMNS = [
    'target', 'status', 'n_phases', 'n_phases_with_pet', 'n_formation_reactions',
    'n_practical_formation_reactions',
    'energy_above_hull', 'local_dft_e_above_hull', 'local_pet_e_above_hull',
    'n_reactions_with_dft_pet', 'n_reactions_with_pet_mp_form',
    'mae_pet_vs_dft', 'rmse_pet_vs_dft', 'bias_pet_vs_dft',
    'mae_pet_vs_mp_form', 'rmse_pet_vs_mp_form', 'bias_pet_vs_mp_form',
    'sign_consistency', 'sign_consistency_mp_form',
    'best_dft_reaction',
    'best_dft_deltaE', 'best_mp_form_reaction', 'best_mp_form_deltaE',
    'best_pet_reaction', 'best_pet_deltaE',
    'top1_match', 'top5_overlap', 'best_practical_pet_reaction',
    'best_practical_pet_deltaE', 'best_practical_dft_reaction',
    'best_practical_dft_deltaE', 'best_practical_mp_form_reaction',
    'best_practical_mp_form_deltaE', 'error',
]


def masked_cmd(cmd: list[str]) -> str:
    masked = []
    hide_next = False
    for token in cmd:
        if hide_next:
            masked.append('***')
            hide_next = False
            continue
        masked.append(token)
        if token == '--mp-api-key':
            hide_next = True
    return '$ ' + ' '.join(masked)


def run(cmd: list[str], keep_going: bool) -> tuple[int, str]:
    print(masked_cmd(cmd), flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout[-2000:], flush=True)
    if proc.returncode != 0:
        print(proc.stderr[-4000:], file=sys.stderr, flush=True)
        if not keep_going:
            raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    return proc.returncode, proc.stdout + '\n' + proc.stderr


def failed_metrics(target: str, log: str) -> dict:
    row = {col: np.nan for col in METRIC_COLUMNS}
    row['target'] = normalize_formula(target)
    row['status'] = 'failed'
    row['n_phases'] = 0
    row['n_phases_with_pet'] = 0
    row['n_formation_reactions'] = 0
    row['n_reactions_with_dft_pet'] = 0
    row['n_reactions_with_pet_mp_form'] = 0
    row['error'] = log.strip().splitlines()[-1] if log.strip() else 'run failed'
    return row


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df:
            return col
    return None


def comparison_metrics(df: pd.DataFrame, ref_col: str, pet_col: str, prefix: str) -> tuple[dict, pd.DataFrame]:
    out = {}
    if not len(df) or not {ref_col, pet_col}.issubset(df.columns):
        out[f'n_reactions_with_pet_{prefix}'] = 0
        return out, pd.DataFrame()
    both = df[df[ref_col].notna() & df[pet_col].notna()].copy()
    out[f'n_reactions_with_pet_{prefix}'] = len(both)
    if not len(both):
        return out, both
    err = both[pet_col] - both[ref_col]
    out[f'mae_pet_vs_{prefix}'] = float(err.abs().mean())
    out[f'rmse_pet_vs_{prefix}'] = float(np.sqrt((err ** 2).mean()))
    out[f'bias_pet_vs_{prefix}'] = float(err.mean())
    out[f'sign_consistency_{prefix}'] = float(((both[ref_col] * both[pet_col]) > 0).mean())
    return out, both


def target_metrics(target_dir: Path, target: str) -> dict:
    meta = {}
    phase_path = target_dir / 'phase_stability.csv'
    formation_path = target_dir / 'formation_reactions.csv'
    report_path = target_dir / 'report.md'
    phase_df = pd.read_csv(phase_path) if phase_path.exists() and phase_path.stat().st_size > 0 else pd.DataFrame()
    formation_df = pd.read_csv(formation_path) if formation_path.exists() and formation_path.stat().st_size > 0 else pd.DataFrame()
    meta['target'] = normalize_formula(target)
    meta['n_phases'] = len(phase_df)
    meta['n_phases_with_pet'] = int(phase_df['PET_energy_per_formula'].notna().sum()) if 'PET_energy_per_formula' in phase_df else 0
    meta['n_formation_reactions'] = len(formation_df)
    practical_df = formation_df[formation_df['practical_route'] == True].copy() if len(formation_df) and 'practical_route' in formation_df else pd.DataFrame()
    meta['n_practical_formation_reactions'] = len(practical_df)
    target_rows = phase_df[phase_df['formula'] == normalize_formula(target)] if len(phase_df) and 'formula' in phase_df else pd.DataFrame()
    if len(target_rows):
        row = target_rows.iloc[0]
        for col in ['energy_above_hull', 'local_dft_e_above_hull', 'local_pet_e_above_hull']:
            meta[col] = row.get(col)
    pet_col = first_existing_column(formation_df, ['DeltaE_PET-OAM-XI_per_target', 'DeltaE_PET-OAM-XI'])
    form_col = first_existing_column(formation_df, ['DeltaE_MP_FORM_per_target', 'DeltaE_MP_FORM', 'DeltaE_DFT_per_target', 'DeltaE_DFT'])
    if pet_col and form_col:
        form_metrics, both = comparison_metrics(formation_df, form_col, pet_col, 'mp_form')
        meta.update(form_metrics)
    else:
        both = pd.DataFrame()
        meta['n_reactions_with_pet_mp_form'] = 0
    meta['n_reactions_with_dft_pet'] = len(both)
    if len(both):
        meta['mae_pet_vs_dft'] = meta.get('mae_pet_vs_mp_form')
        meta['rmse_pet_vs_dft'] = meta.get('rmse_pet_vs_mp_form')
        meta['bias_pet_vs_dft'] = meta.get('bias_pet_vs_mp_form')
        meta['sign_consistency'] = meta.get('sign_consistency_mp_form')
        dft_best = both.nsmallest(1, form_col).iloc[0]
        pet_best = both.nsmallest(1, pet_col).iloc[0]
        meta['best_dft_reaction'] = dft_best['reaction']
        meta['best_dft_deltaE'] = float(dft_best[form_col])
        meta['best_mp_form_reaction'] = dft_best['reaction']
        meta['best_mp_form_deltaE'] = float(dft_best[form_col])
        meta['best_pet_reaction'] = pet_best['reaction']
        meta['best_pet_deltaE'] = float(pet_best[pet_col])
        meta['top1_match'] = bool(dft_best['reaction'] == pet_best['reaction'])
        dft_top5 = set(both.nsmallest(min(5, len(both)), form_col)['reaction'])
        pet_top5 = set(both.nsmallest(min(5, len(both)), pet_col)['reaction'])
        meta['top5_overlap'] = len(dft_top5 & pet_top5)
        if len(practical_df):
            if pet_col in practical_df and practical_df[pet_col].notna().any():
                best = practical_df[practical_df[pet_col].notna()].nsmallest(1, pet_col).iloc[0]
                meta['best_practical_pet_reaction'] = best['reaction']
                meta['best_practical_pet_deltaE'] = float(best[pet_col])
            if form_col in practical_df and practical_df[form_col].notna().any():
                best = practical_df[practical_df[form_col].notna()].nsmallest(1, form_col).iloc[0]
                meta['best_practical_dft_reaction'] = best['reaction']
                meta['best_practical_dft_deltaE'] = float(best[form_col])
                meta['best_practical_mp_form_reaction'] = best['reaction']
                meta['best_practical_mp_form_deltaE'] = float(best[form_col])
    else:
        for col in ['mae_pet_vs_dft', 'rmse_pet_vs_dft', 'bias_pet_vs_dft', 'sign_consistency', 'best_dft_reaction', 'best_dft_deltaE', 'best_pet_reaction', 'best_pet_deltaE', 'top1_match', 'top5_overlap', 'best_practical_pet_reaction', 'best_practical_pet_deltaE', 'best_practical_dft_reaction', 'best_practical_dft_deltaE']:
            meta[col] = np.nan
    return meta


def plot_summary(summary: pd.DataFrame, pooled: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    valid = summary[summary['n_reactions_with_dft_pet'] > 0].sort_values('mae_pet_vs_dft')
    ax = axes[0, 0]
    ax.barh(valid['target'], valid['mae_pet_vs_dft'], color='#009E73')
    ax.invert_yaxis()
    ax.set_xlabel('PET vs MP-formation reaction MAE (eV/target)')
    ax.set_title('Target-wise reaction-energy error')

    ax = axes[0, 1]
    ax.barh(valid['target'], valid['n_reactions_with_dft_pet'], color='#4C78A8')
    ax.invert_yaxis()
    ax.set_xlabel('Comparable reactions')
    ax.set_title('Workflow coverage')

    ax = axes[1, 0]
    if len(pooled):
        dft_col = first_existing_column(pooled, ['DeltaE_MP_FORM_per_target', 'DeltaE_MP_FORM', 'DeltaE_DFT_per_target', 'DeltaE_DFT'])
        pet_col = 'DeltaE_PET-OAM-XI_per_target' if 'DeltaE_PET-OAM-XI_per_target' in pooled else 'DeltaE_PET-OAM-XI'
        ax.scatter(pooled[dft_col], pooled[pet_col], s=12, alpha=0.55, color='#D55E00')
        lo = min(pooled[dft_col].min(), pooled[pet_col].min())
        hi = max(pooled[dft_col].max(), pooled[pet_col].max())
        ax.plot([lo, hi], [lo, hi], color='0.2', lw=1)
    ax.set_xlabel('MP-formation reaction energy per target (eV)')
    ax.set_ylabel('PET reaction energy per target (eV)')
    ax.set_title('Pooled reaction parity')

    ax = axes[1, 1]
    if len(valid):
        ax.scatter(valid['energy_above_hull'], valid['mae_pet_vs_dft'], s=45, color='#9467BD')
        for _, row in valid.iterrows():
            if pd.notna(row.get('energy_above_hull')):
                ax.text(row['energy_above_hull'], row['mae_pet_vs_dft'], row['target'], fontsize=7)
    ax.set_xlabel('MP energy above hull (eV/atom)')
    ax.set_ylabel('PET reaction MAE (eV)')
    ax.set_title('Target stability vs reaction error')

    fig.suptitle('MP-expanded PET synthesis-pathway benchmark', fontsize=15, fontweight='bold')
    fig.savefig(output_dir / 'mp_pet_benchmark_summary.png', dpi=300)
    fig.savefig(output_dir / 'mp_pet_benchmark_summary.pdf')
    plt.close(fig)


def add_category_summaries(summary: pd.DataFrame, output_dir: Path) -> Path | None:
    if summary.empty:
        return None
    categorized = summary.copy()
    categorized['category'] = categorized['target'].map(lambda value: CATEGORY_LOOKUP.get(normalize_formula(value), 'custom/uncategorized'))
    if (categorized['category'] == 'custom/uncategorized').all():
        return None
    categorized.to_csv(output_dir / 'mp_pet_benchmark_summary_with_categories.csv', index=False)
    rows = []
    for category, sub in categorized.groupby('category'):
        valid = sub[sub['n_reactions_with_dft_pet'].fillna(0) > 0]
        rows.append(
            {
                'category': category,
                'n_targets': int(len(sub)),
                'n_phases': int(sub['n_phases'].fillna(0).sum()),
                'n_pet_phases': int(sub['n_phases_with_pet'].fillna(0).sum()),
                'n_formation_reactions': int(sub['n_formation_reactions'].fillna(0).sum()),
                'n_comparable_reactions': int(valid['n_reactions_with_dft_pet'].fillna(0).sum()),
                'n_practical_routes': int(sub['n_practical_formation_reactions'].fillna(0).sum()),
                'mean_mae_pet_vs_mp_form': float(valid['mae_pet_vs_mp_form'].mean()) if len(valid) else np.nan,
                'median_mae_pet_vs_mp_form': float(valid['mae_pet_vs_mp_form'].median()) if len(valid) else np.nan,
                'mean_sign_consistency': float(valid['sign_consistency'].mean()) if len(valid) else np.nan,
                'top1_match_rate': float(valid['top1_match'].mean()) if len(valid) else np.nan,
            }
        )
    category_df = pd.DataFrame(rows).sort_values('category').reset_index(drop=True)
    category_df.to_csv(output_dir / 'mp_pet_benchmark_category_summary.csv', index=False)
    return output_dir / 'mp_pet_benchmark_summary_with_categories.csv'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-set', choices=sorted(TARGET_SETS), default='core25')
    parser.add_argument('--targets', nargs='*', default=None)
    parser.add_argument('--output-dir', default=str(PROJECT / 'results' / 'mp_pet_benchmark'))
    parser.add_argument('--mp-api-key', default=None)
    parser.add_argument('--mp-max-phases', type=int, default=40)
    parser.add_argument('--mp-force-refresh', action='store_true')
    parser.add_argument('--pet-device', default='auto')
    parser.add_argument('--pet-python', default=None)
    parser.add_argument('--pet-max-phases', type=int, default=None)
    parser.add_argument('--max-reactants', type=int, default=2)
    parser.add_argument('--max-byproducts', type=int, default=1)
    parser.add_argument('--max-products', type=int, default=2)
    parser.add_argument('--max-total-coeff', type=int, default=50)
    parser.add_argument('--keep-going', action='store_true')
    args = parser.parse_args()
    targets = args.targets if args.targets else TARGET_SETS[args.target_set]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    pooled_rows = []
    script = str(PROJECT / 'run_screening.py')
    for target in targets:
        target = normalize_formula(target)
        target_dir = output_dir / target
        cmd = [
            '/home/zhzhan/miniconda3/envs/mattersim/bin/python', script,
            '--target-formula', target,
            '--database', 'mp',
            '--mp-max-phases', str(args.mp_max_phases),
            '--compute-pet',
            '--pet-device', args.pet_device,
            '--max-reactants', str(args.max_reactants),
            '--max-byproducts', str(args.max_byproducts),
            '--max-products', str(args.max_products),
            '--max-total-coeff', str(args.max_total_coeff),
            '--output-dir', str(target_dir),
        ]
        if args.mp_api_key:
            cmd.extend(['--mp-api-key', args.mp_api_key])
        if args.mp_force_refresh:
            cmd.append('--mp-force-refresh')
        if args.pet_max_phases:
            cmd.extend(['--pet-max-phases', str(args.pet_max_phases)])
        if args.pet_python:
            cmd.extend(['--pet-python', args.pet_python])
        code, _ = run(cmd, keep_going=args.keep_going)
        if code != 0:
            rows.append(failed_metrics(target, _))
            continue
        metrics = target_metrics(target_dir, target)
        metrics['status'] = 'ok'
        rows.append(metrics)
        formation_path = target_dir / 'formation_reactions.csv'
        if formation_path.exists() and formation_path.stat().st_size > 0:
            df = pd.read_csv(formation_path)
            form_col = first_existing_column(df, ['DeltaE_MP_FORM_per_target', 'DeltaE_MP_FORM', 'DeltaE_DFT_per_target', 'DeltaE_DFT'])
            pet_col = first_existing_column(df, ['DeltaE_PET-OAM-XI_per_target', 'DeltaE_PET-OAM-XI'])
            required = [c for c in [form_col, pet_col] if c]
            if len(required) == 2:
                sub = df[df[form_col].notna() & df[pet_col].notna()].copy()
                sub['target'] = target
                keep_cols = ['target', 'reaction', form_col, pet_col]
                pooled_rows.append(sub[keep_cols])

    summary = pd.DataFrame(rows)
    for col in METRIC_COLUMNS:
        if col not in summary:
            summary[col] = np.nan
    summary = summary[METRIC_COLUMNS]
    summary.to_csv(output_dir / 'mp_pet_benchmark_summary.csv', index=False)
    pooled = pd.concat(pooled_rows, ignore_index=True) if pooled_rows else pd.DataFrame()
    pooled.to_csv(output_dir / 'mp_pet_reaction_data.csv', index=False)
    if len(summary):
        valid = summary[summary['n_reactions_with_dft_pet'].fillna(0) > 0]
        readme = ['# MP-expanded PET benchmark', '']
        readme.append(f'- Targets attempted: {len(summary)}')
        readme.append(f'- Target set: {args.target_set if not args.targets else "custom"}')
        readme.append(f'- Targets with comparable PET/MP-formation reactions: {len(valid)}')
        if len(valid):
            readme.append(f'- Total comparable reactions: {int(valid["n_reactions_with_dft_pet"].sum())}')
            readme.append(f'- Practical-route candidates: {int(valid["n_practical_formation_reactions"].fillna(0).sum())}')
            readme.append(f'- Mean PET vs MP-formation MAE: {valid["mae_pet_vs_mp_form"].mean():.4f} eV/target')
            readme.append(f'- Median PET vs MP-formation MAE: {valid["mae_pet_vs_mp_form"].median():.4f} eV/target')
            readme.append(f'- Mean sign consistency: {valid["sign_consistency"].mean():.3f}')
            readme.append(f'- Top-1 match rate: {valid["top1_match"].mean():.3f}')
        (output_dir / 'README_summary.md').write_text('\n'.join(readme), encoding='utf-8')
        plot_summary(summary, pooled, output_dir)
        category_table = add_category_summaries(summary, output_dir)
        aggregate_recommendations(
            benchmark_dir=output_dir,
            category_table=category_table,
        )
    print(summary.to_string(index=False))
    print(f'Saved benchmark to {output_dir}')


if __name__ == '__main__':
    main()
