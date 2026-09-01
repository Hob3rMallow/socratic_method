[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$submissionRoot = $PSScriptRoot
$repositoryRoot = (Resolve-Path (Join-Path $submissionRoot "..\..")).Path
$sourceFile = Join-Path $submissionRoot "src\paper.tex"
$outputDirectory = Join-Path $submissionRoot "output"
$paperStem = Join-Path $outputDirectory "paper"
$releaseDirectory = Join-Path $repositoryRoot "output\pdf"
$releasePdf = Join-Path $releaseDirectory "socratic_method_siggraph_draft.pdf"

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $releaseDirectory | Out-Null

$latexArguments = @(
    "-interaction=nonstopmode"
    "-halt-on-error"
    "-output-directory=$outputDirectory"
    $sourceFile
)

Push-Location $repositoryRoot
try {
    & pdflatex @latexArguments
    if ($LASTEXITCODE -ne 0) {
        throw "The first pdflatex pass failed with exit code $LASTEXITCODE."
    }

    & bibtex $paperStem
    if ($LASTEXITCODE -ne 0) {
        throw "BibTeX failed with exit code $LASTEXITCODE."
    }

    & pdflatex @latexArguments
    if ($LASTEXITCODE -ne 0) {
        throw "The second pdflatex pass failed with exit code $LASTEXITCODE."
    }

    & pdflatex @latexArguments
    if ($LASTEXITCODE -ne 0) {
        throw "The final pdflatex pass failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

Copy-Item -LiteralPath "$paperStem.pdf" -Destination $releasePdf -Force
Write-Output "Built $paperStem.pdf"
Write-Output "Copied release draft to $releasePdf"
