# GitHub upload commands

The local release folder has been prepared as a Git repository. Because this
computer is not currently authenticated to GitHub, create an empty repository
named `InvDesRoute` on GitHub first, then run:

```powershell
cd C:\Users\86139\Desktop\InvDesRoute_release
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/InvDesRoute.git
git branch -M main
git push -u origin main
```

If Git Credential Manager asks you to log in, use the browser login flow.

After the GitHub repository is online, make a release tag:

```powershell
cd C:\Users\86139\Desktop\InvDesRoute_release
git tag -a v2026.06.26 -m "InvDesRoute manuscript data and code release"
git push origin v2026.06.26
```

Then upload `C:\Users\86139\Desktop\InvDesRoute_release_20260626.tar.gz` to
Zenodo and replace these placeholders in the manuscript:

- `ZENODO_DOI_OR_URL`
- `GITHUB_OR_ARCHIVE_URL`
- optionally `CODE_ARCHIVE_DOI_OR_URL`
