from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition

from common import normalize_formula


def formula_stoich_ratio(formula: str) -> float:
    amounts = list(Composition(formula).as_dict().values())
    if len(amounts) <= 1:
        return 1.0
    return float(max(amounts) / min(amounts))


def parse_reaction_side(text: str) -> list[tuple[str, int]]:
    species = []
    if not isinstance(text, str) or not text:
        return species
    for part in text.split(' | '):
        if '*' not in part:
            continue
        coeff, formula = part.split('*', 1)
        species.append((normalize_formula(formula), int(coeff)))
    return species


def phase_ehull_value(phase: dict) -> float | None:
    for col in ['energy_above_hull', 'local_dft_e_above_hull', 'local_pet_e_above_hull']:
        value = phase.get(col)
        if pd.notna(value):
            return float(value)
    return None


def route_precursor_descriptors(row: pd.Series, target_formula: str, phase_lookup: dict[str, dict]) -> dict:
    target = normalize_formula(target_formula)
    reactants = parse_reaction_side(row.get('reactants', ''))
    products = parse_reaction_side(row.get('products', ''))
    byproducts = [(formula, coeff) for formula, coeff in products if formula != target]
    reactant_formulas = [formula for formula, _ in reactants]
    byproduct_formulas = [formula for formula, _ in byproducts]
    reactant_atoms = [float(Composition(formula).num_atoms) for formula in reactant_formulas]
    reactant_elements = [len(Composition(formula).as_dict()) for formula in reactant_formulas]
    precursor_ehulls = []
    for formula in reactant_formulas:
        value = phase_ehull_value(phase_lookup.get(formula, {}))
        if value is not None:
            precursor_ehulls.append(value)
    max_reactant_atoms = max(reactant_atoms) if reactant_atoms else np.nan
    max_reactant_elements = max(reactant_elements) if reactant_elements else np.nan
    elemental_only = bool(reactant_elements) and max_reactant_elements == 1
    if elemental_only:
        complexity = 'elemental_precursors'
    elif pd.notna(max_reactant_elements) and max_reactant_elements <= 2 and pd.notna(max_reactant_atoms) and max_reactant_atoms <= 10:
        complexity = 'simple_compound_precursors'
    else:
        complexity = 'complex_precursors'
    return {
        'reactant_formulas': ';'.join(reactant_formulas),
        'byproduct_formulas': ';'.join(byproduct_formulas) if byproduct_formulas else 'none',
        'n_reactants': len(reactants),
        'n_byproducts': len(byproducts),
        'max_reactant_atoms': max_reactant_atoms,
        'max_reactant_elements': max_reactant_elements,
        'elemental_reactants_only': elemental_only,
        'precursor_complexity_class': complexity,
        'max_precursor_ehull': max(precursor_ehulls) if precursor_ehulls else np.nan,
    }


def mp_formation_energy_per_formula(row: pd.Series) -> float | None:
    if pd.notna(row.get('formation_energy_per_atom')) and pd.notna(row.get('n_atoms')):
        return float(row['formation_energy_per_atom']) * float(row['n_atoms'])
    return None


def source_dft_energy_per_formula(row: pd.Series) -> float | None:
    if pd.notna(row.get('dft_energy_per_formula')):
        return float(row['dft_energy_per_formula'])
    return None


def mp_formation_energy_map(phase_df: pd.DataFrame, fallback_to_source_energy: bool = False) -> dict[str, float]:
    if phase_df.empty:
        return {}
    out = {}
    for _, row in phase_df.iterrows():
        energy = mp_formation_energy_per_formula(row)
        if energy is None and fallback_to_source_energy:
            energy = source_dft_energy_per_formula(row)
        if energy is not None:
            out[normalize_formula(row['formula'])] = energy
    return out


def compatible_dft_energy_per_formula(row: pd.Series) -> float | None:
    energy = mp_formation_energy_per_formula(row)
    return energy if energy is not None else source_dft_energy_per_formula(row)


def dft_energy_map(phase_df: pd.DataFrame) -> dict[str, float]:
    return mp_formation_energy_map(phase_df, fallback_to_source_energy=True)


def annotate_reaction_practicality(
    reaction_df: pd.DataFrame,
    phase_df: pd.DataFrame,
    target_formula: str,
    max_atoms: int = 10,
    max_stoich_ratio: float = 5.0,
    max_ehull: float = 0.10,
    max_coeff_sum_per_target: float = 30.0,
) -> pd.DataFrame:
    if reaction_df.empty:
        return reaction_df
    df = reaction_df.copy()
    target = normalize_formula(target_formula)
    phase_lookup = phase_df.set_index('formula').to_dict('index') if len(phase_df) and 'formula' in phase_df else {}
    rows = []
    for _, row in df.iterrows():
        species = parse_reaction_side(row.get('reactants', '')) + parse_reaction_side(row.get('products', ''))
        non_target = [(formula, coeff) for formula, coeff in species if formula != target]
        target_coeff = float(row.get('target_coefficient') or 1.0)
        atom_counts = [float(Composition(formula).num_atoms) for formula, _ in non_target] or [0.0]
        ratios = [formula_stoich_ratio(formula) for formula, _ in non_target] or [1.0]
        ehulls = []
        for formula, _ in non_target:
            value = phase_ehull_value(phase_lookup.get(formula, {}))
            if value is not None:
                ehulls.append(value)
        coeff_sum_per_target = sum(coeff for _, coeff in species) / target_coeff
        max_atoms_seen = max(atom_counts)
        max_ratio_seen = max(ratios)
        max_ehull_seen = max(ehulls) if ehulls else np.nan
        practical = (
            max_atoms_seen <= max_atoms
            and max_ratio_seen <= max_stoich_ratio
            and (pd.isna(max_ehull_seen) or max_ehull_seen <= max_ehull)
            and coeff_sum_per_target <= max_coeff_sum_per_target
        )
        rows.append(
            {
                'max_non_target_atoms': max_atoms_seen,
                'max_non_target_stoich_ratio': max_ratio_seen,
                'max_non_target_ehull': max_ehull_seen,
                'coeff_sum_per_target': coeff_sum_per_target,
                'practical_route': practical,
            }
        )
    extra = pd.DataFrame(rows, index=df.index)
    return pd.concat([df, extra], axis=1)


def pet_energy_map(cache_csv: str | Path) -> dict[str, float]:
    path = Path(cache_csv)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    df = pd.read_csv(path)
    if 'PET_energy_per_formula' not in df:
        return {}
    return {normalize_formula(r['formula']): float(r['PET_energy_per_formula']) for _, r in df.iterrows() if pd.notna(r['PET_energy_per_formula'])}


def compute_phase_stability(phase_df: pd.DataFrame, pet_map: dict[str, float] | None = None) -> pd.DataFrame:
    df = phase_df.copy()
    if 'formula' not in df:
        return df
    df['formula'] = df['formula'].map(normalize_formula)
    df['PET_energy_per_formula'] = df['formula'].map(pet_map or {})
    df['PET_energy_per_atom'] = df['PET_energy_per_formula'] / df['n_atoms'] if 'n_atoms' in df else np.nan
    df['compatible_dft_energy_per_formula'] = [
        compatible_dft_energy_per_formula(row) for _, row in df.iterrows()
    ]
    df['mp_formation_energy_per_formula'] = [
        mp_formation_energy_per_formula(row) for _, row in df.iterrows()
    ]

    # MP/API Ehull if available is the most reliable direct value.
    if 'energy_above_hull' not in df:
        df['energy_above_hull'] = np.nan

    # Local convex-hull diagnostics from available DFT energies.
    dft_entries = []
    dft_formula_to_entry = {}
    for _, row in df.iterrows():
        if pd.notna(row.get('compatible_dft_energy_per_formula')):
            entry = PDEntry(Composition(row['formula']), float(row['compatible_dft_energy_per_formula']))
            dft_entries.append(entry)
            dft_formula_to_entry[row['formula']] = entry
    if len(dft_entries) >= 2:
        try:
            pd_dft = PhaseDiagram(dft_entries)
            df['local_dft_e_above_hull'] = df['formula'].map(
                lambda f: pd_dft.get_e_above_hull(dft_formula_to_entry[f]) if f in dft_formula_to_entry else np.nan
            )
        except Exception:
            df['local_dft_e_above_hull'] = np.nan
    else:
        df['local_dft_e_above_hull'] = np.nan

    pet_entries = []
    pet_formula_to_entry = {}
    for _, row in df.iterrows():
        if pd.notna(row.get('PET_energy_per_formula')):
            entry = PDEntry(Composition(row['formula']), float(row['PET_energy_per_formula']))
            pet_entries.append(entry)
            pet_formula_to_entry[row['formula']] = entry
    if len(pet_entries) >= 2:
        try:
            pd_pet = PhaseDiagram(pet_entries)
            df['local_pet_e_above_hull'] = df['formula'].map(
                lambda f: pd_pet.get_e_above_hull(pet_formula_to_entry[f]) if f in pet_formula_to_entry else np.nan
            )
        except Exception:
            df['local_pet_e_above_hull'] = np.nan
    else:
        df['local_pet_e_above_hull'] = np.nan
    return df


def best_paths(formation_df: pd.DataFrame, decomposition_df: pd.DataFrame) -> dict:
    result = {}
    ranking_cols = [
        'DeltaE_PET-OAM-XI_per_target',
        'DeltaE_MP_FORM_per_target',
        'DeltaE_DFT_per_target',
        'DeltaE_PET-OAM-XI',
        'DeltaE_MP_FORM',
        'DeltaE_DFT',
    ]
    for label, df in [('formation', formation_df), ('decomposition', decomposition_df)]:
        for col in ranking_cols:
            key = f'best_{label}_{col.replace("DeltaE_", "")}'
            if len(df) and col in df and df[col].notna().any():
                row = df[df[col].notna()].nsmallest(1, col).iloc[0]
                result[key] = {'reaction': row['reaction'], 'deltaE': float(row[col])}
            else:
                result[key] = None
    return result


def classify(best: dict, target_row: pd.Series | None = None) -> tuple[str, list[str]]:
    reasons = []
    formation = best.get('best_formation_PET-OAM-XI_per_target') or best.get('best_formation_MP_FORM_per_target') or best.get('best_formation_DFT_per_target') or best.get('best_formation_PET-OAM-XI') or best.get('best_formation_MP_FORM') or best.get('best_formation_DFT')
    decomposition = best.get('best_decomposition_PET-OAM-XI_per_target') or best.get('best_decomposition_MP_FORM_per_target') or best.get('best_decomposition_DFT_per_target') or best.get('best_decomposition_PET-OAM-XI') or best.get('best_decomposition_MP_FORM') or best.get('best_decomposition_DFT')
    if not formation:
        return 'insufficient data', ['No formation pathway could be ranked with available energies.']
    if formation['deltaE'] < -0.05:
        reasons.append(f"Best formation pathway is exergonic ({formation['deltaE']:.3f} eV).")
        formation_ok = True
    else:
        reasons.append(f"Best formation pathway is not clearly exergonic ({formation['deltaE']:.3f} eV).")
        formation_ok = False
    risky = False
    if decomposition:
        if decomposition['deltaE'] < -0.05:
            risky = True
            reasons.append(f"Best decomposition pathway is also exergonic ({decomposition['deltaE']:.3f} eV).")
        else:
            reasons.append(f"No strongly exergonic decomposition pathway was found ({decomposition['deltaE']:.3f} eV).")
    else:
        reasons.append('No ranked decomposition pathway was found in the enumerated network.')
    if target_row is not None:
        for col, name in [('energy_above_hull', 'MP Ehull'), ('local_dft_e_above_hull', 'local DFT Ehull'), ('local_pet_e_above_hull', 'local PET Ehull')]:
            if col in target_row and pd.notna(target_row[col]):
                reasons.append(f'{name} for the target is {float(target_row[col]):.4f} eV/atom.')
                break
    if formation_ok and not risky:
        return 'promising thermodynamic pathway', reasons
    if formation_ok and risky:
        return 'competition-limited', reasons
    return 'not promising under screened conditions', reasons


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df:
            return col
    return None


def _best_row(df: pd.DataFrame, col: str | None) -> pd.Series | None:
    if col and len(df) and col in df and df[col].notna().any():
        return df[df[col].notna()].nsmallest(1, col).iloc[0]
    return None


def _top_reactions(df: pd.DataFrame, col: str | None, n: int = 5) -> list[str]:
    if not col or not len(df) or col not in df or not df[col].notna().any():
        return []
    return df[df[col].notna()].nsmallest(min(n, df[col].notna().sum()), col)['reaction'].tolist()


def route_recommendation(
    target: str,
    phase_df: pd.DataFrame,
    formation_df: pd.DataFrame,
    decomposition_df: pd.DataFrame,
    top_n: int = 10,
    exergonic_threshold: float = -0.05,
    ehull_threshold: float = 0.10,
) -> tuple[pd.DataFrame, dict, str]:
    target = normalize_formula(target)
    target_rows = phase_df[phase_df['formula'] == target] if len(phase_df) and 'formula' in phase_df else pd.DataFrame()
    target_row = target_rows.iloc[0] if len(target_rows) else None
    target_ehull = np.nan
    for col in ['energy_above_hull', 'local_dft_e_above_hull', 'local_pet_e_above_hull']:
        if target_row is not None and col in target_row and pd.notna(target_row[col]):
            target_ehull = float(target_row[col])
            break

    pet_col = first_existing_column(formation_df, ['DeltaE_PET-OAM-XI_per_target', 'DeltaE_PET-OAM-XI'])
    ref_col = first_existing_column(formation_df, ['DeltaE_MP_FORM_per_target', 'DeltaE_MP_FORM', 'DeltaE_DFT_per_target', 'DeltaE_DFT'])
    dec_pet_col = first_existing_column(decomposition_df, ['DeltaE_PET-OAM-XI_per_target', 'DeltaE_PET-OAM-XI'])
    dec_ref_col = first_existing_column(decomposition_df, ['DeltaE_MP_FORM_per_target', 'DeltaE_MP_FORM', 'DeltaE_DFT_per_target', 'DeltaE_DFT'])
    has_pet = bool(pet_col and len(formation_df) and pet_col in formation_df and formation_df[pet_col].notna().any())
    rank_col = pet_col if has_pet else ref_col
    rank_label = 'PET-OAM-XI' if has_pet else 'MP_FORM'
    dec_rank_col = dec_pet_col if has_pet and dec_pet_col else dec_ref_col
    phase_lookup = phase_df.set_index('formula').to_dict('index') if len(phase_df) and 'formula' in phase_df else {}

    practical_df = formation_df[formation_df.get('practical_route', False) == True].copy() if len(formation_df) and 'practical_route' in formation_df else pd.DataFrame()
    ranked_source = practical_df if rank_col and len(practical_df) and rank_col in practical_df and practical_df[rank_col].notna().any() else formation_df.copy()
    source_label = 'practical' if ranked_source is practical_df and len(practical_df) else 'all'
    best_pet = _best_row(formation_df, pet_col)
    best_ref = _best_row(formation_df, ref_col)
    comparable_df = pd.DataFrame()
    if pet_col and ref_col and len(formation_df) and {pet_col, ref_col}.issubset(formation_df.columns):
        comparable_df = formation_df[formation_df[pet_col].notna() & formation_df[ref_col].notna()].copy()
    comparable_best_pet = _best_row(comparable_df, pet_col)
    comparable_best_ref = _best_row(comparable_df, ref_col)
    best_practical_ranked = _best_row(practical_df, rank_col)
    best_decomposition_ranked = _best_row(decomposition_df, dec_rank_col)

    top_pet = _top_reactions(comparable_df, pet_col, 5)
    top_ref = _top_reactions(comparable_df, ref_col, 5)
    top1_match = bool(comparable_best_pet is not None and comparable_best_ref is not None and comparable_best_pet.get('reaction') == comparable_best_ref.get('reaction'))
    top5_overlap = len(set(top_pet) & set(top_ref)) if top_pet and top_ref else np.nan

    confidence_score = 0
    risk_flags = []
    if not has_pet:
        risk_flags.append('pet_energy_unavailable_using_mp_form_reference')
    if pd.notna(target_ehull) and target_ehull <= ehull_threshold:
        confidence_score += 1
    else:
        risk_flags.append('target_high_or_unknown_ehull')
    if best_practical_ranked is not None and rank_col and float(best_practical_ranked[rank_col]) < exergonic_threshold:
        confidence_score += 1
    else:
        risk_flags.append(f'no_exergonic_practical_{rank_label.lower()}_route')
    if has_pet and top1_match:
        confidence_score += 1
    elif has_pet and pd.notna(top5_overlap) and top5_overlap >= 3:
        confidence_score += 0.5
        risk_flags.append('top1_mismatch_but_top5_overlap')
    elif not has_pet:
        confidence_score += 0.5
    else:
        risk_flags.append('pet_mpform_pathway_disagreement')
    if best_decomposition_ranked is not None and dec_rank_col and float(best_decomposition_ranked[dec_rank_col]) < exergonic_threshold:
        risk_flags.append('exergonic_decomposition_competition')
    else:
        confidence_score += 1
    if not has_pet:
        confidence_score = min(confidence_score, 2.5)

    if confidence_score >= 3.5:
        confidence = 'high'
    elif confidence_score >= 2.0:
        confidence = 'medium'
    else:
        confidence = 'low'

    if confidence == 'high':
        classification = 'promising synthesis candidate'
    elif confidence == 'medium':
        classification = 'plausible but needs DFT/experimental filtering'
    else:
        classification = 'uncertain under current screened network'

    recommendations = []
    if rank_col and len(ranked_source) and rank_col in ranked_source:
        ranked = ranked_source[ranked_source[rank_col].notna()].nsmallest(top_n, rank_col).copy()
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            descriptors = route_precursor_descriptors(row, target, phase_lookup)
            ref_delta = float(row[ref_col]) if ref_col and ref_col in row and pd.notna(row[ref_col]) else np.nan
            pet_delta = float(row[pet_col]) if pet_col and pet_col in row and pd.notna(row[pet_col]) else np.nan
            ranked_delta = float(row[rank_col])
            agreement = np.nan
            if pd.notna(ref_delta) and pd.notna(pet_delta):
                agreement = abs(pet_delta - ref_delta)
            recommendations.append(
                {
                    'target': target,
                    'rank': rank,
                    'route_scope': source_label,
                    'ranking_energy_source': rank_label,
                    'reaction': row['reaction'],
                    'ranked_deltaE_per_target': ranked_delta,
                    'pet_deltaE_per_target': pet_delta,
                    'mp_form_deltaE_per_target': ref_delta,
                    'abs_pet_mp_form_gap': agreement,
                    'practical_route': bool(row.get('practical_route')) if 'practical_route' in row else False,
                    'max_non_target_ehull': row.get('max_non_target_ehull', np.nan),
                    'coeff_sum_per_target': row.get('coeff_sum_per_target', np.nan),
                    **descriptors,
                    'recommendation_confidence': confidence,
                    'classification': classification,
                    'risk_flags': ';'.join(risk_flags) if risk_flags else 'none',
                }
            )
    rec_df = pd.DataFrame(recommendations)

    summary = {
        'target': target,
        'classification': classification,
        'recommendation_confidence': confidence,
        'confidence_score': confidence_score,
        'risk_flags': ';'.join(risk_flags) if risk_flags else 'none',
        'ranking_energy_source': rank_label,
        'target_ehull': target_ehull,
        'n_phases': int(len(phase_df)),
        'n_formation_reactions': int(len(formation_df)),
        'n_practical_formation_reactions': int(len(practical_df)),
        'n_decomposition_reactions': int(len(decomposition_df)),
        'best_pet_reaction': best_pet.get('reaction') if best_pet is not None else None,
        'best_pet_deltaE_per_target': float(best_pet[pet_col]) if best_pet is not None else np.nan,
        'best_mp_form_reaction': comparable_best_ref.get('reaction') if comparable_best_ref is not None else (best_ref.get('reaction') if best_ref is not None else None),
        'best_mp_form_deltaE_per_target': float(comparable_best_ref[ref_col]) if comparable_best_ref is not None else (float(best_ref[ref_col]) if best_ref is not None else np.nan),
        'best_practical_ranked_reaction': best_practical_ranked.get('reaction') if best_practical_ranked is not None else None,
        'best_practical_ranked_deltaE_per_target': float(best_practical_ranked[rank_col]) if best_practical_ranked is not None and rank_col else np.nan,
        'best_decomposition_ranked_reaction': best_decomposition_ranked.get('reaction') if best_decomposition_ranked is not None else None,
        'best_decomposition_ranked_deltaE_per_target': float(best_decomposition_ranked[dec_rank_col]) if best_decomposition_ranked is not None and dec_rank_col else np.nan,
        'top1_pet_mp_form_match': top1_match,
        'top5_pet_mp_form_overlap': top5_overlap,
    }

    lines = [
        f'# Recommended synthesis routes: {target}',
        '',
        f'- Classification: **{classification}**',
        f'- Recommendation confidence: **{confidence}** (score = {confidence_score:.1f}/4)',
        f'- Ranking energy source: **{rank_label}**',
        f'- Risk flags: `{summary["risk_flags"]}`',
        f'- Target Ehull: `{target_ehull:.6f}` eV/atom' if pd.notna(target_ehull) else '- Target Ehull: unavailable',
        f'- Formation reactions: `{len(formation_df)}`; practical formation reactions: `{len(practical_df)}`',
        f'- PET/MP_FORM top-1 match: `{top1_match}`; top-5 overlap: `{top5_overlap}`',
        '',
        f'## Best practical {rank_label} route',
    ]
    if best_practical_ranked is not None and rank_col:
        lines.append(f'- `{best_practical_ranked["reaction"]}`')
        lines.append(f'- {rank_label} ΔE/target = `{float(best_practical_ranked[rank_col]):.6f}` eV')
        if rank_label != 'PET-OAM-XI' and pet_col and pet_col in best_practical_ranked and pd.notna(best_practical_ranked[pet_col]):
            lines.append(f'- PET ΔE/target = `{float(best_practical_ranked[pet_col]):.6f}` eV')
        if rank_label != 'MP_FORM' and ref_col and ref_col in best_practical_ranked and pd.notna(best_practical_ranked[ref_col]):
            lines.append(f'- MP_FORM ΔE/target = `{float(best_practical_ranked[ref_col]):.6f}` eV')
    else:
        lines.append(f'- No practical {rank_label}-ranked route is available.')
    lines.extend(['', '## Top recommended routes'])
    if len(rec_df):
        for _, row in rec_df.head(top_n).iterrows():
            ref_value = row['mp_form_deltaE_per_target']
            ref_text = f', MP_FORM {ref_value:.3f} eV' if row['ranking_energy_source'] != 'MP_FORM' and pd.notna(ref_value) else ''
            pet_value = row['pet_deltaE_per_target']
            pet_text = f', PET {pet_value:.3f} eV' if row['ranking_energy_source'] != 'PET-OAM-XI' and pd.notna(pet_value) else ''
            lines.append(f'- {int(row["rank"])}. `{row["reaction"]}` ; {row["ranking_energy_source"]} {row["ranked_deltaE_per_target"]:.3f} eV{pet_text}{ref_text}')
    else:
        lines.append(f'- No route could be ranked with {rank_label} energies.')
    lines.extend(['', '## Interpretation'])
    if confidence == 'high':
        lines.append(f'- The target has a low Ehull, an exergonic practical {rank_label} route, route-ranking support, and no strongly exergonic decomposition route in the screened network.')
    elif confidence == 'medium':
        lines.append('- The target has at least partial thermodynamic support, but one or more risk flags should be checked before treating the top route as robust.')
    else:
        lines.append('- The current phase network does not provide enough consistent thermodynamic evidence for a confident recommendation.')
    lines.append(f'- Recommended next step: run PET if unavailable, then DFT refinement for the best practical {rank_label} route and any MP_FORM top-ranked route if they differ.')
    return rec_df, summary, '\n'.join(lines)


def write_recommendation(
    outdir: str | Path,
    target: str,
    phase_df: pd.DataFrame,
    formation_df: pd.DataFrame,
    decomposition_df: pd.DataFrame,
    exergonic_threshold: float = -0.05,
    ehull_threshold: float = 0.10,
) -> None:
    outdir = Path(outdir)
    rec_df, summary, markdown = route_recommendation(
        target,
        phase_df,
        formation_df,
        decomposition_df,
        exergonic_threshold=exergonic_threshold,
        ehull_threshold=ehull_threshold,
    )
    rec_df.to_csv(outdir / 'recommended_routes.csv', index=False)
    pd.DataFrame([summary]).to_csv(outdir / 'synthesis_recommendation_summary.csv', index=False)
    (outdir / 'synthesis_recommendation.md').write_text(markdown, encoding='utf-8')


def write_report(outdir: str | Path, target: str, phase_df: pd.DataFrame, formation_df: pd.DataFrame, decomposition_df: pd.DataFrame) -> None:
    outdir = Path(outdir)
    best = best_paths(formation_df, decomposition_df)
    target_rows = phase_df[phase_df['formula'] == normalize_formula(target)] if len(phase_df) else pd.DataFrame()
    target_row = target_rows.iloc[0] if len(target_rows) else None
    cls, reasons = classify(best, target_row)
    lines = [
        f'# Synthesis pathway screening report: {normalize_formula(target)}',
        '',
        f'- Classification: **{cls}**',
        f'- Number of phases: `{len(phase_df)}`',
        f'- Formation reactions: `{len(formation_df)}`',
        f'- Decomposition reactions: `{len(decomposition_df)}`',
        '',
        '## Target stability diagnostics',
    ]
    if target_row is not None:
        for col in ['energy_above_hull', 'local_dft_e_above_hull', 'local_pet_e_above_hull']:
            if col in target_row and pd.notna(target_row[col]):
                lines.append(f'- `{col}` = {float(target_row[col]):.6f} eV/atom')
    else:
        lines.append('- Target not present in the final phase table.')
    lines.extend(['', '## Best formation pathways'])
    for col in ['DeltaE_PET-OAM-XI_per_target', 'DeltaE_MP_FORM_per_target', 'DeltaE_DFT_per_target']:
        if len(formation_df) and col in formation_df and formation_df[col].notna().any():
            row = formation_df[formation_df[col].notna()].nsmallest(1, col).iloc[0]
            lines.append(f'- `{col}`: `{row["reaction"]}` ; ΔE/target = {float(row[col]):.6f} eV')
    if not any(line.startswith('- `DeltaE') for line in lines[-2:]):
        lines.append('- No ranked formation pathways available.')
    lines.extend(['', '## Best practical formation pathways'])
    practical_formation = formation_df[formation_df.get('practical_route', False) == True] if len(formation_df) and 'practical_route' in formation_df else pd.DataFrame()
    wrote = False
    for col in ['DeltaE_PET-OAM-XI_per_target', 'DeltaE_MP_FORM_per_target', 'DeltaE_DFT_per_target']:
        if len(practical_formation) and col in practical_formation and practical_formation[col].notna().any():
            row = practical_formation[practical_formation[col].notna()].nsmallest(1, col).iloc[0]
            lines.append(f'- `{col}`: `{row["reaction"]}` ; ΔE/target = {float(row[col]):.6f} eV')
            wrote = True
    if not wrote:
        lines.append('- No low-complexity, low-Ehull practical formation route passed the current filters.')
    lines.extend(['', '## Best decomposition pathways'])
    wrote = False
    for col in ['DeltaE_PET-OAM-XI_per_target', 'DeltaE_MP_FORM_per_target', 'DeltaE_DFT_per_target']:
        if len(decomposition_df) and col in decomposition_df and decomposition_df[col].notna().any():
            row = decomposition_df[decomposition_df[col].notna()].nsmallest(1, col).iloc[0]
            lines.append(f'- `{col}`: `{row["reaction"]}` ; ΔE/target = {float(row[col]):.6f} eV')
            wrote = True
    if not wrote:
        lines.append('- No ranked decomposition pathways available.')
    lines.extend(['', '## Best practical decomposition pathways'])
    practical_decomposition = decomposition_df[decomposition_df.get('practical_route', False) == True] if len(decomposition_df) and 'practical_route' in decomposition_df else pd.DataFrame()
    wrote = False
    for col in ['DeltaE_PET-OAM-XI_per_target', 'DeltaE_MP_FORM_per_target', 'DeltaE_DFT_per_target']:
        if len(practical_decomposition) and col in practical_decomposition and practical_decomposition[col].notna().any():
            row = practical_decomposition[practical_decomposition[col].notna()].nsmallest(1, col).iloc[0]
            lines.append(f'- `{col}`: `{row["reaction"]}` ; ΔE/target = {float(row[col]):.6f} eV')
            wrote = True
    if not wrote:
        lines.append('- No low-complexity, low-Ehull practical decomposition route passed the current filters.')
    lines.extend(['', '## Interpretation'])
    for reason in reasons:
        lines.append(f'- {reason}')
    (outdir / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
