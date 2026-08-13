param(
    [string]$Python = "python"
)

$ErrorActionPreference = 'Stop'

Write-Host "--- Starting Build ---"

# All paths are relative to the script location (no hardcoding needed)
$RepoRoot = $PSScriptRoot
$SpecFile = Join-Path $RepoRoot "CMSmark.spec"
$DistDir  = Join-Path $RepoRoot "dist"
$BuildDir = Join-Path $RepoRoot "build"

Write-Host "Project Root : $RepoRoot"
Write-Host "Spec File    : $SpecFile"

# Verify models and icon are present
$modelsPath = Join-Path $RepoRoot "src\models"
if (-not (Test-Path $modelsPath)) {
    Write-Warning "Models folder not found at '$modelsPath'. The build may fail."
}
$iconFile = Join-Path $RepoRoot "icon\app_icon.ico"
if (-not (Test-Path $iconFile)) {
    Write-Warning "Icon file not found at '$iconFile'. The .exe will build without an icon."
}

# --- Step 1: Clean previous artifacts ---
Write-Host "[1/3] Cleaning previous build artifacts..."
if (Test-Path $DistDir)  { Remove-Item -Recurse -Force $DistDir }
if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
# Also clean any stale artifacts accidentally created inside src/
$SrcBuildDir = Join-Path $RepoRoot "src\build"
$SrcDistDir  = Join-Path $RepoRoot "src\dist"
if (Test-Path $SrcBuildDir) { Remove-Item -Recurse -Force $SrcBuildDir }
if (Test-Path $SrcDistDir)  { Remove-Item -Recurse -Force $SrcDistDir }
Write-Host "Cleaning complete."

# --- Step 2: Build using the spec file (single source of truth for what gets bundled) ---
Write-Host "[2/3] Building with PyInstaller (spec file)..."
& $Python -m PyInstaller $SpecFile --noconfirm --clean
if ($LASTEXITCODE -ne 0) {
    Write-Error "--- Build Failed (exit code $LASTEXITCODE) ---"
    exit $LASTEXITCODE
}

Write-Host "[3/3] Done." -ForegroundColor Green
Write-Host "--- Build Succeeded ---" -ForegroundColor Green
Write-Host "Executable: $(Join-Path $DistDir 'CMSMARK.exe')"
