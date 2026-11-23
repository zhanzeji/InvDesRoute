# Data sources and redistribution

Materials Project structures and thermodynamic metadata originate from https://materialsproject.org/ and remain subject to its applicable terms. The archive records identifiers, structures and processed tables; it does not assert a new license over third-party data.

Literature sources include Kononova et al., https://doi.org/10.1038/s41597-019-0224-1 and https://github.com/CederGroupHub/text-mined-synthesis_public ; and Lee et al., https://doi.org/10.1038/s41597-025-06222-y with data https://doi.org/10.6084/m9.figshare.30423274 . Source text-mining inaccuracies and quality flags are retained; extracted expressions are not all independently certified balanced reactions.

Gas thermal tables are from NIST-JANAF, https://janaf.nist.gov/ , DOI https://doi.org/10.18434/T42S31 . Raw table snapshots and processing code are included. DOI publication metadata were resolved using OpenAlex/Crossref; those records are not experimental energy measurements.

Model sources, revisions and hashes are recorded in the archive. Model checkpoints are not included. Do not apply the author-code license to model weights or upstream implementations. Before publishing the data record, the responsible authors must confirm the redistribution scope and choose a suitable record-level license; a compilation license does not override original source terms.

## v0.1.0 redistribution review

| Material | Archive location | Release decision pending |
| --- | --- | --- |
| Materials Project structures and thermodynamic inputs | `data/structures/`, `data/structure_manifest.csv` and applicable input tables | Map archived records to source dataset and version. The [MP database release notes](https://docs.materialsproject.org/changes/database-versions) identify GNoME structures with BY-NC terms. Confirm the terms and attribution for the actual records included. |
| Kononova and Lee literature records | `data/literature/references/` and `data/literature/source/SS_rxns_80806.json.gz` | Confirm the dataset-specific redistribution terms, including the full Lee source snapshot. The [Lee dataset DOI](https://doi.org/10.6084/m9.figshare.30423274) was located in the paper, but the deposit license could not be verified during this review. |
| NIST-JANAF reference tables | Seven raw snapshots under `revision/source_snapshots/` in the software archive, plus derived data tables in the data archive | Confirm SRD 13 terms for verbatim snapshots before releasing them. See the [JANAF record](https://janaf.nist.gov/) and [NIST SRD rights guidance](https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications). |
| Author-controlled processed outputs | Processed results, split manifests and validation tables, subject to file-level provenance | The authors approved CC BY 4.0 for the four processed outputs listed below. |

If redistribution terms for upstream snapshots remain unresolved, cite the original repositories and have reproducibility scripts obtain those inputs separately. Verify the analyses against the resulting package before using the final Data Availability statement. 

The public data ZIP includes the per-target core metrics, seven-model metrics, external endpoint summary and archived validation flags. These author-generated outputs are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Obtain the third-party inputs from their sources above. For gas analysis, retrieve the seven tables listed in `GASES` in `revision/tools/gas_thermochemistry.py` from JANAF and place them under `revision/source_snapshots/` using the file names expected by `read_table`. The full archived benchmark inputs are required for end-to-end recomputation.
