$ErrorActionPreference = "Continue"

Write-Host "============================================="
Write-Host " WhyLab V3 Ultimate Overhaul Deployment      "
Write-Host "============================================="

Write-Host "`n[1/3] Compiling LaTeX to PDF..."
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
cd ..

Write-Host "`n[2/3] Building Submission ZIP Packages..."
python build_submission.py
python scripts/package_anonymous.py

Write-Host "`n[3/3] Committing and Pushing to 3 GitHub Remotes (V3)..."
git add -A
git commit -m "feat: Apply V3 Ultimate Overhaul for NeurIPS Strong Accept (SAHOO baseline, Docker execution, Multi-model stats)"

Write-Host "-> Yesol-Pilot/WhyLab (origin)"
git push origin HEAD --force

Write-Host "-> neogenesislab (neogenesis)"
git push neogenesis HEAD --force

Write-Host "-> openreview-neurips (neurips)"
git push neurips HEAD --force

Write-Host "`n============================================="
Write-Host " Deployment V3 Complete! "
Write-Host "============================================="
