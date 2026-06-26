from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from common import normalize_formula
from analysis import parse_reaction_side


GAS_RESERVOIRS = {normalize_formula(formula) for formula in ['O2', 'N2', 'H2', 'NH3', 'CO2', 'CO', 'H2O', 'Cl2', 'F2', 'Br2', 'I2']}
ELEMENTAL_GASES = {normalize_formula(formula) for formula in ['O2', 'N2', 'H2', 'Cl2', 'F2', 'Br2', 'I2']}
COMMON_BYPRODUCTS = {normalize_formula(formula) for formula in ['CO2', 'CO', 'H2O', 'H2', 'O2', 'N2', 'NH3']}
HALOGENS = {'F', 'Cl', 'Br', 'I'}
ALKALI = {'Li', 'Na', 'K', 'Rb', 'Cs'}
ALKALINE_EARTH = {'Mg', 'Ca', 'Sr', 'Ba'}
TRANSITION_METALS = {'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Hf', 'Ta', 'W'}
COMMON_EXPERIMENTAL_OXIDES = {
    normalize_formula(formula)
    for formula in [
        'Li2O',
        'Li2O2',
        'Na2O',
        'K2O',
        'MgO',
        'CaO',
        'SrO',
        'BaO',
        'Al2O3',
        'SiO2',
        'TiO2',
        'V2O5',
        'Cr2O3',
        'MnO',
        'MnO2',
        'Mn2O3',
        'Mn3O4',
        'FeO',
        'Fe2O3',
        'Fe3O4',
        'CoO',
        'Co3O4',
        'NiO',
        'CuO',
        'Cu2O',
        'ZnO',
        'ZrO2',
        'Nb2O5',
        'MoO3',
        'WO3',
        'P2O5',
    ]
}


def element_count(formula: str) -> int:
    return len(Composition(formula).as_dict())


def formula_atoms(formula: str) -> int:
    return int(Composition(formula).num_atoms)


def is_elemental(formula: str) -> bool:
    return element_count(formula) == 1


def contains_any(formula: str, elements: set[str]) -> bool:
    return bool(set(Composition(formula).as_dict()) & elements)


def stoich_ratio(formula: str) -> float:
    amounts = list(Composition(formula).as_dict().values())
    if len(amounts) <= 1:
        return 1.0
    return float(max(amounts) / min(amounts))


def is_binary_oxide(formula: str) -> bool:
    formula = normalize_formula(formula)
    elements = set(Composition(formula).as_dict())
    return 'O' in elements and len(elements) == 2


def is_solid_binary_oxide(formula: str) -> bool:
    formula = normalize_formula(formula)
    elements = set(Composition(formula).as_dict())
    if formula in GAS_RESERVOIRS:
        return False
    if elements <= {'H', 'C', 'N', 'O'}:
        return False
    return 'O' in elements and len(elements) == 2


def is_common_experimental_oxide(formula: str) -> bool:
    return normalize_formula(formula) in COMMON_EXPERIMENTAL_OXIDES


def is_carbonate_like(formula: str) -> bool:
    elements = set(Composition(formula).as_dict())
    return 'C' in elements and 'O' in elements and bool(elements - {'C', 'O'})


def is_nitrate_or_ammonium_like(formula: str) -> bool:
    elements = set(Composition(formula).as_dict())
    return 'N' in elements and ('O' in elements or 'H' in elements) and bool(elements - {'N', 'O', 'H'})


def is_nitrogen_oxide_or_oxoacid(formula: str) -> bool:
    formula = normalize_formula(formula)
    elements = set(Composition(formula).as_dict())
    if formula in {normalize_formula('N2'), normalize_formula('NH3')}:
        return False
    return {'N', 'O'}.issubset(elements) and elements.issubset({'H', 'N', 'O'})


def is_oxalate_like(formula: str) -> bool:
    comp = Composition(formula).as_dict()
    return 'C' in comp and 'O' in comp and comp.get('O', 0) >= 2 * comp.get('C', 0) and bool(set(comp) - {'C', 'O'})


def is_unusual_halogen_nitrogen(formula: str) -> bool:
    elements = set(Composition(formula).as_dict())
    return 'N' in elements and bool(elements & HALOGENS) and elements.issubset({'N', *HALOGENS})


def precursor_class(formula: str, source: str | None = None) -> str:
    formula = normalize_formula(formula)
    if formula in GAS_RESERVOIRS:
        return 'gas_reservoir'
    if is_unusual_halogen_nitrogen(formula):
        return 'reactive_halogen_nitrogen_precursor'
    if stoich_ratio(formula) > 8:
        return 'unusual_high_stoichiometry_precursor'
    if source == 'virtual_common_precursor':
        return 'virtual_precursor'
    comp = Composition(formula)
    elements = set(comp.as_dict())
    if len(elements) == 1:
        return 'elemental_solid'
    if 'C' in elements and 'O' in elements and len(elements) >= 3:
        return 'carbonate_or_organic_salt'
    if 'N' in elements and 'O' in elements and len(elements) >= 3:
        return 'nitrate_or_ammonium_salt'
    if 'H' in elements and 'O' in elements:
        return 'hydroxide_or_hydrate'
    if is_solid_binary_oxide(formula):
        if is_common_experimental_oxide(formula):
            return 'common_binary_oxide'
        return 'binary_oxide'
    if len(elements) == 2 and (elements & {'F', 'Cl', 'Br', 'I', 'S', 'N', 'P'}):
        return 'binary_salt'
    return 'compound_precursor'


def precursor_prior(formula: str, source: str | None = None) -> tuple[float, str]:
    cls = precursor_class(formula, source)
    if cls == 'reactive_halogen_nitrogen_precursor':
        return 0.18, 'halogen-nitrogen precursor is highly reactive and poor as a first experimental suggestion'
    if cls == 'unusual_high_stoichiometry_precursor':
        return 0.25, 'unusual high-stoichiometry computed phase is unlikely as a practical precursor'
    if cls == 'common_binary_oxide':
        return 0.95, 'binary oxide is a common solid-state precursor'
    if cls == 'binary_oxide':
        return 0.52, 'binary oxide appears in the computed phase pool but is less common as a first experimental precursor'
    if cls == 'carbonate_or_organic_salt':
        return 0.90, 'carbonate/organic salt is a common ceramic precursor'
    if cls == 'hydroxide_or_hydrate':
        return 0.82, 'hydroxide/hydrate is experimentally accessible but may require dehydration'
    if cls == 'nitrate_or_ammonium_salt':
        return 0.78, 'nitrate/ammonium salt is common but decomposition gases matter'
    if cls == 'binary_salt':
        return 0.75, 'binary salt is a common precursor family'
    if cls == 'elemental_solid':
        return 0.68, 'elemental solid is simple but can be air-sensitive/reactive'
    if cls == 'gas_reservoir':
        return 0.62, 'gas reservoir is experimentally controllable but condition-dependent'
    if cls == 'virtual_precursor':
        return 0.55, 'precursor is chemically plausible but not present in MP; thermochemical data are missing'
    return 0.58, 'compound precursor has moderate experimental plausibility'


def route_energy_score(delta: float | None, exergonic_threshold: float = -0.05) -> tuple[float, str]:
    if delta is None or pd.isna(delta):
        return 0.35, 'reaction energy unavailable; route requires manual/DFT thermochemical completion'
    value = float(delta)
    if value <= -1.0:
        return 1.00, 'strong thermodynamic driving force'
    if value <= exergonic_threshold:
        return 0.88, 'exergonic formation route'
    if value <= 0.25:
        return 0.64, 'near-thermoneutral route; feasible under favorable conditions'
    if value <= 1.0:
        return 0.42, 'mildly uphill static energy; may require finite-temperature or gas corrections'
    return 0.20, 'strongly uphill static energy; low thermodynamic support without corrections'


def route_complexity_score(reactants: list[str], byproducts: list[str], coeff_sum_per_target: float | None) -> tuple[float, str]:
    n_reactants = len(reactants)
    n_byproducts = len(byproducts)
    max_atoms = max([formula_atoms(formula) for formula in reactants + byproducts], default=1)
    coeff_value = float(coeff_sum_per_target) if coeff_sum_per_target is not None and pd.notna(coeff_sum_per_target) else np.nan
    score = 1.0
    reasons = []
    if n_reactants <= 2:
        reasons.append('few reactants')
    elif n_reactants == 3:
        score -= 0.12
        reasons.append('three-reactant route')
    else:
        score -= 0.25
        reasons.append('many-reactant route')
    if n_byproducts == 0:
        reasons.append('no byproduct bookkeeping')
    elif n_byproducts <= 2:
        score -= 0.08
        reasons.append('limited byproduct set')
    else:
        score -= 0.22
        reasons.append('many byproducts; gas evolution/decomposition must be managed')
    if max_atoms > 20:
        score -= 0.20
        reasons.append('large precursor formula')
    elif max_atoms > 10:
        score -= 0.08
        reasons.append('moderately complex precursor formula')
    if pd.notna(coeff_value):
        if coeff_value > 80:
            score -= 0.25
            reasons.append('large stoichiometric coefficient sum')
        elif coeff_value > 40:
            score -= 0.12
            reasons.append('moderate stoichiometric coefficient sum')
    return max(0.05, min(1.0, score)), '; '.join(reasons) if reasons else 'simple route'


def route_condition_score(reactants: list[str], byproducts: list[str]) -> tuple[float, str, list[str]]:
    gases = sorted((set(reactants) | set(byproducts)) & GAS_RESERVOIRS)
    flags = []
    score = 1.0
    if gases:
        score -= 0.10
        flags.append('gas_reservoir_or_gas_release')
    if any(formula in byproducts for formula in ['CO2', 'CO', 'H2O', 'H2', 'H3N']):
        score -= 0.08
        flags.append('gas_byproduct_management')
    if any(formula in reactants for formula in ['O2', 'N2', 'H2', 'H3N']):
        score -= 0.05
        flags.append('atmosphere_control_needed')
    if any(contains_any(formula, {'C', 'H', 'N'}) for formula in reactants + byproducts):
        score -= 0.05
        flags.append('finite_temperature_decomposition_likely_relevant')
    reason = 'solid-state route with limited atmosphere complexity' if not flags else '; '.join(flags)
    return max(0.10, min(1.0, score)), reason, flags


def route_template_score(target: str, reactants: list[str], byproducts: list[str]) -> tuple[float, str, list[str]]:
    target_elements = set(Composition(target).as_dict())
    reactant_set = set(reactants)
    byproduct_set = set(byproducts)
    reactant_elements = set().union(*(set(Composition(formula).as_dict()) for formula in reactants)) if reactants else set()
    flags = []
    has_carbonate = any(is_carbonate_like(formula) for formula in reactants)
    has_oxide = any(is_solid_binary_oxide(formula) for formula in reactants)
    has_nitrate = any(is_nitrate_or_ammonium_like(formula) and 'O' in set(Composition(formula).as_dict()) for formula in reactants)
    has_nitrogen_oxide = any(is_nitrogen_oxide_or_oxoacid(formula) for formula in reactants)
    has_reducing_gas = bool(reactant_set & {normalize_formula('CO'), normalize_formula('H2')})
    has_oxalate = any(is_oxalate_like(formula) for formula in reactants)
    has_ammonium_phosphate = any({'H', 'N', 'P', 'O'}.issubset(set(Composition(formula).as_dict())) for formula in reactants)
    has_transition_oxide = any(is_solid_binary_oxide(formula) and bool(set(Composition(formula).as_dict()) & TRANSITION_METALS) for formula in reactants)
    has_li_carbonate = normalize_formula('Li2CO3') in reactant_set
    has_nh3 = normalize_formula('NH3') in reactant_set
    has_co2 = normalize_formula('CO2') in reactant_set or normalize_formula('CO2') in byproduct_set
    target_is_oxide = 'O' in target_elements
    target_is_lithium_transition_oxide = target_is_oxide and 'Li' in target_elements and bool(target_elements & TRANSITION_METALS)
    target_is_binary_oxide = target_is_oxide and len(target_elements) == 2
    target_is_nitride = 'N' in target_elements and len(target_elements) >= 2
    target_is_carbonate = 'C' in target_elements and 'O' in target_elements
    target_is_phosphate = 'P' in target_elements and 'O' in target_elements
    target_is_binary_carbide = 'C' in target_elements and len(target_elements) == 2
    target_is_binary_sulfide = 'S' in target_elements and len(target_elements) == 2
    target_is_binary_halide = bool(target_elements & HALOGENS) and len(target_elements) == 2
    if target_is_phosphate and has_li_carbonate and has_oxalate and has_ammonium_phosphate:
        return 1.00, 'classic lithium phosphate route: carbonate + oxalate + ammonium phosphate', ['classic_phosphate_precursor_route']
    if target_is_binary_oxide:
        metal = next(iter(target_elements - {'O'}), None)
        if metal and normalize_formula(metal) in reactant_set and normalize_formula('O2') in reactant_set:
            return 0.90, 'direct oxidation route: element + O2 -> binary oxide', ['direct_oxidation_route']
        if has_carbonate and normalize_formula('CO2') in byproduct_set:
            return 0.72, 'carbonate/decomposition route to binary oxide', ['binary_oxide_decomposition_route']
    if target_is_binary_carbide:
        metal = next(iter(target_elements - {'C'}), None)
        if metal and normalize_formula(metal) in reactant_set and normalize_formula('C') in reactant_set:
            return 0.86, 'direct carburization route: element + carbon -> carbide', ['direct_carburization_route']
    if target_is_binary_sulfide:
        metal = next(iter(target_elements - {'S'}), None)
        if metal and normalize_formula(metal) in reactant_set and normalize_formula('S') in reactant_set:
            return 0.88, 'direct sulfidation route: element + sulfur -> sulfide', ['direct_sulfidation_route']
    if target_is_binary_halide:
        metal = next(iter(target_elements - HALOGENS), None)
        halogen_reservoirs = {normalize_formula(formula) for formula in ['F2', 'Cl2', 'Br2', 'I2', 'F', 'Cl', 'Br', 'I']}
        if metal and normalize_formula(metal) in reactant_set and bool(reactant_set & halogen_reservoirs):
            return 0.88, 'direct halogenation route: element + halogen reservoir -> halide', ['direct_halogenation_route']
    if target_is_lithium_transition_oxide and has_li_carbonate and has_transition_oxide and normalize_formula('CO2') in byproduct_set:
        return 1.00, 'classic battery-oxide route: lithium carbonate + transition-metal oxide with gas release', ['classic_battery_carbonate_route']
    if target_is_lithium_transition_oxide and has_carbonate and has_transition_oxide and normalize_formula('CO2') in byproduct_set:
        return 0.82, 'alternative carbonate/oxide route for lithium transition-metal oxide', ['nonstandard_lithium_source']
    if target_is_oxide and has_carbonate and has_oxide and normalize_formula('CO2') in byproduct_set:
        return 1.00, 'classic ceramic route: carbonate + oxide -> target oxide + CO2', ['classic_carbonate_oxide_route']
    if target_is_nitride and has_nh3 and normalize_formula('H2') in byproduct_set:
        return 0.93, 'classic ammonolysis/nitridation route: metal + NH3 -> nitride + H2', ['classic_ammonolysis_route']
    if target_is_nitride and (has_nitrate or has_nitrogen_oxide) and has_reducing_gas:
        return 0.28, 'redox nitrate/reducing-gas nitride route is chemically complex and not a first recommendation', ['redox_nitrate_reducing_gas_route']
    if target_is_carbonate and normalize_formula('CO2') in reactant_set and has_oxide:
        return 0.92, 'classic carbonation route: oxide + CO2 -> carbonate', ['classic_carbonation_route']
    if target_is_oxide and all(is_solid_binary_oxide(formula) for formula in reactants):
        if all(is_common_experimental_oxide(formula) for formula in reactants):
            return 0.82, 'all-oxide solid-state route using common oxide precursors', ['all_oxide_route']
        return 0.58, 'all-oxide route using less common computed oxide precursors', ['computed_oxide_route']
    if target_is_nitride and normalize_formula('N2') in reactant_set:
        return 0.72, 'direct nitridation route using N2', ['direct_nitridation_route']
    if all(is_elemental(formula) or formula in ELEMENTAL_GASES for formula in reactants):
        return 0.68, 'direct elemental combination route', ['direct_elemental_route']
    if has_co2 or bool(reactant_elements & {'C', 'H', 'N'}):
        flags.append('condition_dependent_precursor_decomposition_route')
        return 0.62, 'condition-dependent precursor decomposition route', flags
    return 0.50, 'no strong experimental route template recognized', ['no_template_match']


def experimental_guidance(template_reason: str, reactants: list[str], byproducts: list[str], risk_flags: list[str]) -> dict:
    reactant_set = set(reactants)
    byproduct_set = set(byproducts)
    if 'carbonate + oxide' in template_reason:
        mode = 'ceramic solid-state synthesis from carbonate/oxide powders'
        operations = 'dry and grind precursor powders; pre-calcine to release CO2; regrind; final calcination/sintering in air or selected oxygen activity'
        checks = 'monitor mass loss and CO2 release; verify phase purity by XRD after intermediate and final calcination; check carbonate residue'
    elif 'lithium carbonate + transition-metal oxide' in template_reason:
        mode = 'battery/cathode-type solid-state synthesis'
        operations = 'mix lithium carbonate with transition-metal oxide; use slight lithium excess if volatilization is expected; calcine in air/O2; regrind and anneal'
        checks = 'check lithium loss, oxidation state, oxygen activity and secondary oxide phases; compare air versus oxygen annealing'
    elif 'ammonolysis' in template_reason:
        mode = 'ammonolysis or nitridation under flowing NH3'
        operations = 'use metal/oxide precursor under NH3 flow; ramp slowly; ensure H2/NH3 exhaust handling; compare NH3 and N2 nitridation windows'
        checks = 'check incomplete nitridation, metal droplets/oxide residues and safety of H2-containing exhaust'
    elif 'carbonation' in template_reason:
        mode = 'carbonation route using CO2 reservoir'
        operations = 'react oxide/hydroxide precursor with CO2-containing atmosphere; control humidity and CO2 partial pressure; use mild-to-moderate temperature'
        checks = 'verify carbonate formation by XRD/IR/TGA; check reversibility or decomposition on heating'
    elif 'all-oxide' in template_reason:
        mode = 'oxide solid-state synthesis'
        operations = 'mix oxide powders stoichiometrically; calcine with intermediate grinding; tune oxygen partial pressure if multiple oxidation states exist'
        checks = 'screen calcination temperature, oxygen activity and competing oxide phases'
    elif 'direct nitridation' in template_reason:
        mode = 'direct nitridation route'
        operations = 'react metal or precursor under N2/NH3 at elevated temperature; use controlled atmosphere and slow heating'
        checks = 'check kinetic limitations of N2 activation and residual metallic/oxide phases'
    elif 'direct oxidation' in template_reason:
        mode = 'direct oxidation under controlled oxygen activity'
        operations = 'react elemental metal or reduced precursor under air/O2; tune oxygen partial pressure and heating rate when multiple oxide valence states are possible'
        checks = 'check over-oxidation/reduction products, mass gain and phase purity by XRD; compare air and controlled-O2 atmospheres for multivalent metals'
    elif 'direct halogenation' in template_reason:
        mode = 'direct halogenation using a halogen reservoir'
        operations = 'combine metal with halogen source under sealed or flow conditions; use corrosion-resistant containment and slow dosing for reactive halogens'
        checks = 'check volatility, moisture sensitivity, residual metal and halogen overpressure safety before scale-up'
    elif 'direct carburization' in template_reason:
        mode = 'high-temperature carburization or carbothermal synthesis'
        operations = 'mix metal/oxide precursor with carbon source; heat under inert, vacuum or reducing atmosphere; adjust carbon activity to suppress secondary carbides'
        checks = 'check residual free carbon, oxide residues and competing carbide stoichiometries by XRD/Raman/elemental analysis'
    elif 'direct sulfidation' in template_reason:
        mode = 'direct sulfidation or chalcogenization route'
        operations = 'react metal or oxide precursor with sulfur/chalcogen reservoir in sealed ampoule or controlled flow; ramp gradually to manage vapor pressure'
        checks = 'check sulfur loss, non-stoichiometry and competing sulfide phases; verify containment compatibility'
    elif 'carbonate/decomposition route to binary oxide' in template_reason:
        mode = 'thermal decomposition or oxidation of carbonate/salt precursor'
        operations = 'calcine carbonate or decomposable salt under air/O2; use intermediate grinding and hold steps to remove volatile byproducts'
        checks = 'track CO2 release, residual carbonate and oxide valence state; compare air/O2/inert atmosphere when redox competition exists'
    elif 'direct elemental' in template_reason:
        mode = 'direct elemental combination'
        operations = 'handle reactive elements under inert atmosphere if needed; use sealed tube or controlled atmosphere; heat gradually'
        checks = 'check volatility, side reactions, air/moisture sensitivity and safety of halogen/chalcogen reservoirs'
    else:
        mode = 'hypothesis-generation route requiring manual chemistry review'
        operations = 'inspect precursor availability and safety; verify stoichiometry; consider replacing unusual computed phases with common experimental precursors'
        checks = 'perform DFT/manual thermochemistry check before experimental prioritization'

    refinement = []
    if 'uphill_static_energy' in risk_flags:
        refinement.append('evaluate finite-temperature free energy or gas chemical-potential correction')
    if 'gas_byproduct_management' in risk_flags or byproduct_set & COMMON_BYPRODUCTS:
        refinement.append('check gas release and open/closed-system assumptions')
    if 'atmosphere_control_needed' in risk_flags or reactant_set & ELEMENTAL_GASES:
        refinement.append('screen atmosphere or chemical potential window')
    if 'virtual_precursor_energy_missing' in risk_flags or 'energy_unavailable' in risk_flags:
        refinement.append('add audited precursor thermochemistry or DFT energy')
    if 'unusual_or_hazardous_precursor' in risk_flags:
        refinement.append('replace unusual/hazardous precursor with safer experimental analogue')
    if not refinement:
        refinement.append('validate top route by DFT or small-scale experimental screening')

    return {
        'suggested_synthesis_mode': mode,
        'suggested_experimental_operations': operations,
        'suggested_experimental_checks': checks,
        'recommended_refinement_actions': '; '.join(dict.fromkeys(refinement)),
    }


def phase_availability_score(reactants: list[str], phase_lookup: dict[str, dict]) -> tuple[float, str, list[str]]:
    sources = [str(phase_lookup.get(formula, {}).get('source', 'missing')) for formula in reactants]
    flags = []
    if any(source == 'missing' for source in sources):
        flags.append('missing_precursor_phase')
    if any(source == 'virtual_common_precursor' for source in sources):
        flags.append('virtual_precursor_energy_missing')
    if any(source == 'manual_reservoir' for source in sources):
        flags.append('manual_reservoir_energy')
    if not flags:
        return 1.0, 'all reactants have database or manual phase entries', flags
    penalty = 0.18 * ('virtual_precursor_energy_missing' in flags) + 0.10 * ('manual_reservoir_energy' in flags) + 0.30 * ('missing_precursor_phase' in flags)
    return max(0.25, 1.0 - penalty), '; '.join(flags), flags


@dataclass(frozen=True)
class RecommendationWeights:
    energy: float = 0.18
    precursor: float = 0.25
    template: float = 0.25
    practical: float = 0.12
    complexity: float = 0.08
    phase_availability: float = 0.07
    condition: float = 0.05


def select_energy(row: pd.Series, preferred_sources: list[str] | None = None) -> tuple[float | None, str]:
    preferred_sources = preferred_sources or ['PET-OAM-XI', 'PET_FORM_ALIGNED', 'MP_FORM', 'THERMO_FORM', 'DFT']
    for source in preferred_sources:
        col = f'DeltaE_{source}_per_target'
        if col in row and pd.notna(row[col]):
            return float(row[col]), source
    for col in ['DeltaE_PET-OAM-XI_per_target', 'DeltaE_PET_FORM_ALIGNED_per_target', 'DeltaE_MP_FORM_per_target', 'DeltaE_THERMO_FORM_per_target', 'DeltaE_DFT_per_target']:
        if col in row and pd.notna(row[col]):
            return float(row[col]), col.replace('DeltaE_', '').replace('_per_target', '')
    return None, 'unavailable'


def score_route(row: pd.Series, target_formula: str, phase_lookup: dict[str, dict], weights: RecommendationWeights | None = None) -> dict:
    weights = weights or RecommendationWeights()
    target = normalize_formula(target_formula)
    reactants = [formula for formula, _ in parse_reaction_side(row.get('reactants', ''))]
    products = [formula for formula, _ in parse_reaction_side(row.get('products', ''))]
    byproducts = [formula for formula in products if formula != target]
    energy, energy_source = select_energy(row)
    energy_score, energy_reason = route_energy_score(energy)
    precursor_scores = []
    precursor_notes = []
    precursor_classes = []
    for formula in reactants:
        source = str(phase_lookup.get(formula, {}).get('source', 'missing'))
        score, note = precursor_prior(formula, source)
        precursor_scores.append(score)
        precursor_notes.append(f'{formula}: {note}')
        precursor_classes.append(precursor_class(formula, source))
    precursor_score = float(np.mean(precursor_scores)) if precursor_scores else 0.0
    template_score, template_reason, template_flags = route_template_score(target, reactants, byproducts)
    if template_score >= 0.95 and (set(byproducts) & COMMON_BYPRODUCTS):
        if energy is None or pd.isna(energy):
            energy_score = max(energy_score, 0.50)
            energy_reason += '; experimentally common gas-release route but thermochemical data are incomplete'
        elif float(energy) <= 3.0:
            energy_score = max(energy_score, 0.60)
            energy_reason += '; classic gas-release route, so finite-temperature corrections may improve favorability'
    complexity_score, complexity_reason = route_complexity_score(reactants, byproducts, row.get('coeff_sum_per_target'))
    condition_score, condition_reason, condition_flags = route_condition_score(reactants, byproducts)
    availability_score, availability_reason, availability_flags = phase_availability_score(reactants, phase_lookup)
    practical_value = bool(row.get('practical_route')) if 'practical_route' in row else True
    practical_score = 1.0 if practical_value else 0.35
    score = (
        weights.energy * energy_score
        + weights.precursor * precursor_score
        + weights.template * template_score
        + weights.practical * practical_score
        + weights.complexity * complexity_score
        + weights.phase_availability * availability_score
        + weights.condition * condition_score
    )
    if template_score >= 0.95:
        score += 0.06
    elif template_score >= 0.90:
        score += 0.03
    score = min(1.0, score)
    risk_flags = []
    risk_flags.extend(template_flags if template_score < 0.70 else [])
    risk_flags.extend(condition_flags)
    risk_flags.extend(availability_flags)
    if energy is None or energy_source == 'unavailable':
        risk_flags.append('energy_unavailable')
    elif energy > 0.25:
        risk_flags.append('uphill_static_energy')
    if not practical_value:
        risk_flags.append('fails_original_practical_filter')
    if any(precursor_class(formula, str(phase_lookup.get(formula, {}).get('source', 'missing'))) in {'reactive_halogen_nitrogen_precursor', 'unusual_high_stoichiometry_precursor'} for formula in reactants):
        risk_flags.append('unusual_or_hazardous_precursor')
    if 'redox_nitrate_reducing_gas_route' in template_flags:
        risk_flags.append('redox_nitrate_reducing_gas_route')
    strengths = []
    if energy is not None and energy <= -0.05:
        strengths.append('thermodynamically favorable')
    if precursor_score >= 0.80:
        strengths.append('uses common experimental precursor families')
    if template_score >= 0.90:
        strengths.append(template_reason)
    if complexity_score >= 0.80:
        strengths.append('low route complexity')
    if availability_score >= 0.90:
        strengths.append('precursor entries are available')
    if not strengths:
        strengths.append('enumerated as a chemically balanced candidate')
    reason = (
        f'Energy: {energy_reason}. Precursors: {"; ".join(precursor_notes) if precursor_notes else "none"}. '
        f'Complexity: {complexity_reason}. Availability: {availability_reason}. Conditions: {condition_reason}.'
    )
    guidance = experimental_guidance(template_reason, reactants, byproducts, risk_flags)
    return {
        'synth_score': score,
        'energy_score': energy_score,
        'precursor_prior_score': precursor_score,
        'route_complexity_score': complexity_score,
        'route_template_score': template_score,
        'practical_filter_score': practical_score,
        'phase_availability_score': availability_score,
        'condition_simplicity_score': condition_score,
        'synth_rank_energy_source': energy_source,
        'synth_rank_deltaE_per_target': energy if energy is not None else np.nan,
        'reactant_classes': ';'.join(precursor_classes),
        'route_template_reason': template_reason,
        'synthesis_strengths': ';'.join(dict.fromkeys(strengths)),
        'synthesis_risk_flags': ';'.join(dict.fromkeys(risk_flags)) if risk_flags else 'none',
        'synthesis_recommendation_reason': reason,
        **guidance,
    }


def add_synthesizability_scores(formation_df: pd.DataFrame, phase_df: pd.DataFrame, target_formula: str) -> pd.DataFrame:
    if formation_df.empty:
        return formation_df
    target = normalize_formula(target_formula)
    phase_df = phase_df.copy()
    if len(phase_df) and 'formula' in phase_df:
        phase_df['formula'] = phase_df['formula'].map(normalize_formula)
    phase_lookup = phase_df.set_index('formula').to_dict('index') if len(phase_df) and 'formula' in phase_df else {}
    rows = [score_route(row, target, phase_lookup) for _, row in formation_df.iterrows()]
    scored = pd.concat([formation_df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    scored = scored.sort_values(['synth_score', 'synth_rank_deltaE_per_target', 'reaction'], ascending=[False, True, True], na_position='last').reset_index(drop=True)
    scored.insert(0, 'synth_rank', range(1, len(scored) + 1))
    return scored


def target_ehull(phase_df: pd.DataFrame, target_formula: str) -> float | None:
    target = normalize_formula(target_formula)
    if phase_df.empty or 'formula' not in phase_df:
        return None
    df = phase_df.copy()
    df['formula'] = df['formula'].map(normalize_formula)
    rows = df[df['formula'] == target]
    if rows.empty:
        return None
    row = rows.iloc[0]
    for col in ['energy_above_hull', 'local_pet_e_above_hull', 'local_dft_e_above_hull']:
        if col in row and pd.notna(row[col]):
            return float(row[col])
    return None


def classify_target_synthesizability(scored: pd.DataFrame, phase_df: pd.DataFrame, target_formula: str) -> dict[str, object]:
    if scored.empty:
        return {
            'target': normalize_formula(target_formula),
            'synthesis_confidence': 'low',
            'synthesis_decision': 'insufficient_route_evidence',
            'synthesis_confidence_score': 0.0,
            'target_ehull': np.nan,
            'best_route': '',
            'best_route_rank': '',
            'best_route_deltaE_per_target': np.nan,
            'best_route_energy_source': '',
            'primary_support': 'no balanced formation route was retained',
            'primary_risks': 'no route evidence',
            'recommended_next_actions': 'expand phase/precursor pool and verify target formula/CIF',
        }
    ehull = target_ehull(phase_df, target_formula)
    top = scored.iloc[0]
    top_flags = [flag for flag in str(top.get('synthesis_risk_flags', '')).split(';') if flag and flag != 'none']
    top_score = float(top.get('synth_score', 0.0))
    delta = top.get('synth_rank_deltaE_per_target')
    delta_value = None if pd.isna(delta) else float(delta)
    n_good = int((scored['synth_score'] >= 0.90).sum()) if 'synth_score' in scored else 0
    n_exergonic = int((scored['synth_rank_deltaE_per_target'] <= -0.05).sum()) if 'synth_rank_deltaE_per_target' in scored else 0
    has_classic_template = bool(float(top.get('route_template_score', 0.0)) >= 0.90)
    low_ehull = ehull is None or ehull <= 0.10
    major_flags = {
        'energy_unavailable',
        'virtual_precursor_energy_missing',
        'missing_precursor_phase',
        'unusual_or_hazardous_precursor',
        'redox_nitrate_reducing_gas_route',
    }
    medium_flags = {
        'uphill_static_energy',
        'gas_byproduct_management',
        'manual_reservoir_energy',
        'atmosphere_control_needed',
        'finite_temperature_decomposition_likely_relevant',
    }
    confidence_score = 0.0
    confidence_score += min(top_score, 1.0) * 45.0
    confidence_score += 15.0 if has_classic_template else 5.0
    confidence_score += 15.0 if low_ehull else -10.0
    confidence_score += min(n_good, 5) * 3.0
    confidence_score += min(n_exergonic, 5) * 2.0
    confidence_score -= 18.0 * bool(set(top_flags) & major_flags)
    confidence_score -= 6.0 * len(set(top_flags) & medium_flags)
    confidence_score = max(0.0, min(100.0, confidence_score))
    if confidence_score >= 78 and not (set(top_flags) & major_flags):
        confidence = 'high'
        decision = 'promising_route_candidate'
    elif confidence_score >= 55:
        confidence = 'medium'
        decision = 'plausible_but_refinement_needed'
    else:
        confidence = 'low'
        decision = 'weak_or_incomplete_synthesis_support'
    support = []
    if has_classic_template:
        support.append(str(top.get('route_template_reason')))
    if delta_value is not None and delta_value <= -0.05:
        support.append('top route is exergonic or thermodynamically favorable on the selected energy scale')
    elif delta_value is not None and delta_value <= 0.25:
        support.append('top route is near thermoneutral and may be condition-assisted')
    if low_ehull:
        support.append('target Ehull is low or unavailable but not flagged high')
    if n_good:
        support.append(f'{n_good} high-score route(s) retained')
    risks = top_flags if top_flags else ['none']
    actions = []
    if set(top_flags) & {'energy_unavailable', 'virtual_precursor_energy_missing', 'missing_precursor_phase'}:
        actions.append('add audited precursor thermochemistry or DFT/PET energy')
    if set(top_flags) & {'gas_byproduct_management', 'manual_reservoir_energy', 'atmosphere_control_needed', 'uphill_static_energy'}:
        actions.append('apply finite-temperature/gas chemical-potential correction')
    if 'unusual_or_hazardous_precursor' in top_flags or 'redox_nitrate_reducing_gas_route' in top_flags:
        actions.append('replace unusual or hazardous precursor with common experimental analogue')
    if not actions:
        actions.append('validate by DFT refinement or small-scale experimental screening')
    return {
        'target': normalize_formula(target_formula),
        'synthesis_confidence': confidence,
        'synthesis_decision': decision,
        'synthesis_confidence_score': round(confidence_score, 3),
        'target_ehull': np.nan if ehull is None else ehull,
        'best_route': top.get('reaction', ''),
        'best_route_rank': top.get('synth_rank', ''),
        'best_route_deltaE_per_target': delta_value if delta_value is not None else np.nan,
        'best_route_energy_source': top.get('synth_rank_energy_source', ''),
        'best_route_template': top.get('route_template_reason', ''),
        'n_high_score_routes_ge_0p90': n_good,
        'n_exergonic_routes': n_exergonic,
        'primary_support': '; '.join(dict.fromkeys(support)) if support else 'balanced route exists but support is limited',
        'primary_risks': '; '.join(dict.fromkeys(risks)),
        'recommended_next_actions': '; '.join(dict.fromkeys(actions)),
    }


def write_synthesizability_outputs(outdir, target_formula: str, phase_df: pd.DataFrame, formation_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    from pathlib import Path

    outdir = Path(outdir)
    scored = add_synthesizability_scores(formation_df, phase_df, target_formula)
    if scored.empty:
        pd.DataFrame().to_csv(outdir / 'synthesizability_ranked_routes.csv', index=False)
        (outdir / 'synthesizability_recommendation.md').write_text('# Synthesizability recommendation\n\nNo formation routes available.\n', encoding='utf-8')
        return scored
    scored.to_csv(outdir / 'synthesizability_ranked_routes.csv', index=False)
    target = normalize_formula(target_formula)
    decision = classify_target_synthesizability(scored, phase_df, target)
    pd.DataFrame([decision]).to_csv(outdir / 'synthesizability_decision.csv', index=False)
    lines = [
        f'# 可合成性推荐：{target}',
        '',
        '该推荐层不只按最低反应能排序，而是综合静态反应能、实验前驱体常见性、路线复杂度、相/能量可用性和气体/副产物风险。',
        '',
        '## 目标级判断',
        '',
        f'- 置信等级：`{decision["synthesis_confidence"]}`；判断：`{decision["synthesis_decision"]}`；分数：`{decision["synthesis_confidence_score"]}/100`',
        f'- 主要支撑：{decision["primary_support"]}',
        f'- 主要风险：`{decision["primary_risks"]}`',
        f'- 建议下一步：{decision["recommended_next_actions"]}',
        '',
        '## Top 可合成性路线',
        '',
    ]
    for _, row in scored.head(top_n).iterrows():
        delta = row.get('synth_rank_deltaE_per_target')
        delta_text = 'unavailable' if pd.isna(delta) else f'{float(delta):.3f} eV/target'
        lines.append(f'- {int(row["synth_rank"])}. `{row["reaction"]}`')
        lines.append(f'  - 综合可合成性分数：`{float(row["synth_score"]):.3f}`；能量来源：`{row["synth_rank_energy_source"]}`；ΔE：`{delta_text}`')
        lines.append(f'  - 推荐理由：{row["synthesis_strengths"]}')
        lines.append(f'  - 风险标记：`{row["synthesis_risk_flags"]}`')
        lines.append(f'  - 建议实验模式：{row["suggested_synthesis_mode"]}')
        lines.append(f'  - 建议检查：{row["recommended_refinement_actions"]}')
    lines.extend(
        [
            '',
            '## 使用方式',
            '',
            '- 高分路线适合作为优先实验假设；如果同时具有放热/近热中性反应能、常见前驱体和低复杂度，优先级最高。',
            '- 带有 `uphill_static_energy`、`gas_byproduct_management` 或 `virtual_precursor_energy_missing` 的路线不应直接否定，而应进入温度、气氛、前驱体分解热或 DFT 精修。',
            '- 该层是可合成性建议，不等同于完整实验工艺；温度、压力、保温时间和气氛窗口将在后续条件模型中补充。',
        ]
    )
    (outdir / 'synthesizability_recommendation.md').write_text('\n'.join(lines), encoding='utf-8')
    return scored
