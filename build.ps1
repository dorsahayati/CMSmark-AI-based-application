param(
    [string]$Name = "CMSMARK",
    [switch]$OneDir,
    [string]$Python = "python"
)

$ErrorActionPreference = 'Stop'

# --- Debugging: Show initial parameters ---
Write-Host "--- Starting Build ---"
Write-Host "Name: $Name"
Write-Host "OneDir: $OneDir"
Write-Host "Python executable: $Python"
Write-Host "----------------------"

# Get the absolute path of the project root by using the script's own location
$RepoRoot = $PSScriptRoot
$CodePath = Join-Path $RepoRoot "src\main.py"
$DistDir = Join-Path $RepoRoot "dist"
$BuildDir = Join-Path $RepoRoot "build"

Write-Host "Project Root: $RepoRoot"
Write-Host "Main Script Path: $CodePath"

# --- Step 1: Clean previous artifacts ---
Write-Host "[1/4] Cleaning previous build artifacts..."
if (Test-Path $DistDir) {
    Write-Host "Removing existing dist folder..."
    Remove-Item -Recurse -Force $DistDir
}
if (Test-Path $BuildDir) {
    Write-Host "Removing existing build folder..."
    Remove-Item -Recurse -Force $BuildDir
}
Write-Host "Cleaning complete."

# --- Step 2: Prepare PyInstaller arguments ---
Write-Host "[2/4] Preparing PyInstaller arguments..."
# On Windows, --add-data uses src;dest. Ensure you have a 'src/models' folder.
$modelsPath = Join-Path $RepoRoot "src\models"
if (-not (Test-Path $modelsPath)) {
    Write-Warning "Models folder not found at '$modelsPath'. The build may fail or the app may crash."
    # We will still attempt to build without it.
}
$addDataModels = "$modelsPath;models"

# Add the icon folder
$iconPath = Join-Path $RepoRoot "icon"
$iconFile = Join-Path $iconPath "app_icon.ico"
if (-not (Test-Path $iconFile)) {
    Write-Warning "Icon file not found at '$iconFile'. The app icon will be missing."
}
$addDataIcon = "$iconPath;icon"

# Path to the src directory
$srcPath = Join-Path $RepoRoot "src"

# Common arguments for PyInstaller
$commonArgs = @(
    "--noconfirm",
    "--name", $Name,
    "--add-data", $addDataModels,
    "--add-data", $addDataIcon,
    "--paths", $srcPath,
    "--collect-submodules", "sklearn",
    "--icon", $iconFile,  # <-- Add this line
    "--windowed"
)

# --- Step 3: Build command ---
Write-Host "[3/4] Building with PyInstaller..."
$buildCmd = if ($OneDir) { "python -m PyInstaller --onedir" } else { "python -m PyInstaller --onefile" }
Write-Host "Command: $buildCmd @commonArgs `"$CodePath`""

# --- Step 4: Execute PyInstaller ---
try {
    if ($OneDir) {
        python -m PyInstaller --onedir @commonArgs "$CodePath"
    } else {
        python -m PyInstaller --onefile @commonArgs "$CodePath"
    }
    Write-Host "--- Build Succeeded ---" -ForegroundColor Green
} catch {
    Write-Error "--- Build Failed ---"
    Write-Error "An error occurred during the PyInstaller execution: $_"
    exit 1
}