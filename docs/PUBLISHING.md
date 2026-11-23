# Publishing the repository and archive

## Before the first public release

Review `release-config.json`: author names/order and affiliation, GitHub owner, version, code license and data license. The preparation does not supply fake identifiers. Select the code license and place its full text in `LICENSE`; the existing `LICENSE.md` explains the pending decision. Confirm the data redistribution scope separately.

Create an account login locally: `gh auth login`. For command-line Zenodo deposit, create a personal access token with deposit write/actions permissions and export `ZENODO_ACCESS_TOKEN` in the current shell. Do not save it in tracked files.

## GitHub

From the repository root, run `python scripts/publish_github.py --owner YOUR_ACCOUNT`. It validates release metadata, then creates a public repository and pushes the prepared code. It will stop on an existing Git checkout rather than altering an unrelated repository. Git author identity must already be configured. This is a user-invoked publication action, not something executed during local preparation.

For updates, use normal Git commits and a versioned release after validating data/model provenance. Keep processed data in its separate archive, not in Git history.

## Zenodo DOI

`python scripts/zenodo_deposit.py --code-zip /path/to/InvDesRoute-code-v0.1.0.zip --data-zip /path/to/InvDesRoute-data-v0.1.0.zip` creates separate code/data drafts, uploads files and requests reserved DOIs. It saves draft IDs and URLs under ignored `publish_state/`. A reserved DOI does not mean public data are already available. Inspect both drafts on Zenodo, verify authors, licenses and files, then publish there. Only published records should be described in the manuscript as available.

The script intentionally does not publish a record irreversibly without that final human review. If a workflow instead uses the GitHub–Zenodo integration, enable the repository in Zenodo before creating the GitHub release; follow Zenodo's current guidance. Do not mix manual and automatic deposits without checking for duplicate records.

After editing metadata or licenses, run `python scripts/build_archives.py` with the data installed to regenerate both archives before uploading. Code and data license values must correspond to valid Zenodo license identifiers (for example, `mit` or `cc-by-4.0` only if appropriate and approved).

## After publication

Record the actual repository URL, code DOI and data DOI in release-config.json and CITATION.cff. Update docs/AVAILABILITY.tex with those real values. Archive the final published code version, not an uncommitted working tree. Save the archive SHA256 and the commit/release tag with the manuscript source.

Official instructions: https://cli.github.com/manual/gh_repo_create ; https://developers.zenodo.org/ ; https://help.zenodo.org/ .
