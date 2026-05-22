#Requires -Version 5.1
<#
.SYNOPSIS
    One-shot setup script for RoboMME benchmark training on Windows.

.DESCRIPTION
    Checks prerequisites, installs uv if missing, creates the Python venv,
    applies required patches (mplib stub + ManiSkill Vulkan parser fix),
    and runs a smoke test to confirm everything works.

.EXAMPLE
    # Run from the project root in PowerShell:
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\setup_windows.ps1

.EXAMPLE
    # Skip the smoke test:
    .\setup_windows.ps1 -SkipSmokeTest
#>

param(
    [switch]$SkipSmokeTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Helpers ────────────────────────────────────────────────────────────────────

function Write-Step  { param($msg) Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-OK    { param($msg) Write-Host "    [OK] $msg" -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "    [WARN] $msg" -ForegroundColor Yellow }
function Write-Fail  { param($msg) Write-Host "`n[FAIL] $msg" -ForegroundColor Red; exit 1 }

$ProjectRoot = $PSScriptRoot
$Venv        = Join-Path $ProjectRoot ".venv"
$Python      = Join-Path $Venv "Scripts\python.exe"

# ── Step 1: Verify we're in the right directory ────────────────────────────────

Write-Step "Checking project directory"
if (-not (Test-Path (Join-Path $ProjectRoot "pyproject.toml"))) {
    Write-Fail "pyproject.toml not found. Run this script from the robomme_benchmark project root."
}
Write-OK "Project root: $ProjectRoot"

# ── Step 2: Check NVIDIA driver / GPU ─────────────────────────────────────────

Write-Step "Checking NVIDIA GPU"
try {
    $smi = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $smi) {
        Write-OK "GPU found: $($smi.Trim())"
    } else {
        Write-Warn "nvidia-smi returned no output. Vulkan GPU rendering may not work."
    }
} catch {
    Write-Fail "nvidia-smi not found. An NVIDIA GPU with driver 527+ is required."
}

# ── Step 3: Check Git ──────────────────────────────────────────────────────────

Write-Step "Checking Git"
try {
    $gitv = & git --version 2>$null
    Write-OK $gitv
} catch {
    Write-Fail "Git not found. Install from https://git-scm.com/download/win"
}

# ── Step 4: Install uv if missing ─────────────────────────────────────────────

Write-Step "Checking uv package manager"
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCmd) {
    Write-Host "    uv not found — installing..." -ForegroundColor Yellow
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        # Refresh PATH for this session
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","User") + ";" + $env:PATH
        $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
        if (-not $uvCmd) {
            Write-Fail "uv installation succeeded but 'uv' is still not on PATH. Open a new PowerShell window and re-run this script."
        }
    } catch {
        Write-Fail "Failed to install uv: $_`nInstall manually: https://docs.astral.sh/uv/getting-started/installation/"
    }
}
Write-OK "uv $(& uv --version)"

# ── Step 5: Create venv and install all dependencies ──────────────────────────

Write-Step "Creating virtual environment and installing dependencies"
Write-Host "    (This downloads ~4 GB on first run — please wait)" -ForegroundColor DarkGray
Set-Location $ProjectRoot

& uv sync
if ($LASTEXITCODE -ne 0) {
    Write-Fail "uv sync failed. Check your internet connection and pyproject.toml."
}
Write-OK "Virtual environment ready at .venv\"

# ── Step 6: Verify CUDA torch was installed (not CPU-only) ────────────────────

Write-Step "Verifying PyTorch CUDA build"
$torchVersion = & $Python -c "import torch; print(torch.__version__)" 2>$null
if ($torchVersion -notmatch "\+cu") {
    Write-Warn "torch $torchVersion appears to be CPU-only. Expected 2.9.1+cu128."
    Write-Warn "Run 'uv lock' then 'uv sync' to regenerate the lockfile with CUDA sources."
} else {
    Write-OK "torch $torchVersion"
}

$cudaAvail = & $Python -c "import torch; print(torch.cuda.is_available())" 2>$null
if ($cudaAvail -ne "True") {
    Write-Warn "torch.cuda.is_available() = $cudaAvail. GPU training will fall back to CPU."
} else {
    Write-OK "CUDA available"
}

# ── Step 7: Create mplib stub (no Windows wheels) ─────────────────────────────

Write-Step "Installing mplib stub (Linux-only package)"
$mplibDir = Join-Path $Venv "Lib\site-packages\mplib"
New-Item -ItemType Directory -Force $mplibDir | Out-Null

$mplibStub = @'
"""
Stub mplib for Windows. mplib has no Windows wheels.
All benchmark tasks work without it; FK motion planning is unavailable.
"""

class Planner:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "mplib is not available on Windows. "
            "Motion planning features require Linux. "
            "Benchmark evaluation without the FK planner still works."
        )
'@
Set-Content -Path (Join-Path $mplibDir "__init__.py") -Value $mplibStub -Encoding UTF8
Write-OK "mplib stub created"

# ── Step 8: Patch ManiSkill Vulkan device parser ───────────────────────────────

Write-Step "Patching ManiSkill Vulkan PCI address parser"
$backendPath = Join-Path $Venv "Lib\site-packages\mani_skill\envs\utils\system\backend.py"

if (-not (Test-Path $backendPath)) {
    Write-Warn "backend.py not found at expected path — skipping patch."
    Write-Warn "If you see Vulkan PCI address errors at runtime, apply the patch manually (see TRAINING_SETUP_WINDOWS.md)."
} else {
    $content = Get-Content $backendPath -Raw

    # Only patch if not already patched
    if ($content -notmatch "startswith\(..pci:") {
        $oldFn = @'
def parse_backend_device_id(backend: str) -> tuple[str, int]:
    if ":" in backend:
        return backend.split(":")
    return backend, None
'@
        $newFn = @'
def parse_backend_device_id(backend: str) -> tuple[str, int]:
    if backend.startswith("pci:"):
        return backend, None  # PCI bus address — keep whole string as backend name
    if ":" in backend:
        return backend.split(":")
    return backend, None
'@
        if ($content -match [regex]::Escape('def parse_backend_device_id')) {
            $content = $content -replace [regex]::Escape($oldFn), $newFn
            Set-Content -Path $backendPath -Value $content -Encoding UTF8 -NoNewline
            Write-OK "backend.py patched"
        } else {
            Write-Warn "Could not find parse_backend_device_id in backend.py — function signature may have changed."
        }
    } else {
        Write-OK "backend.py already patched"
    }
}

# ── Step 9: Smoke test ─────────────────────────────────────────────────────────

if (-not $SkipSmokeTest) {
    Write-Step "Running environment smoke test (loads BinFill, steps once)"
    Write-Host "    (First run reconfigures the scene — takes ~30 seconds)" -ForegroundColor DarkGray

    $smokeScript = @'
import sys, warnings
warnings.filterwarnings("ignore")
from train.envs.rl_env import RobommeRLEnv
env = RobommeRLEnv("BinFill", seed=0)
obs, _ = env.reset()
shapes = {k: list(v.shape) for k, v in obs.items()}
assert shapes["front_rgb"] == [128, 128, 3], f"unexpected front_rgb shape: {shapes['front_rgb']}"
obs2, rew, term, trunc, info = env.step(env.action_space.sample())
env.close()
print("SMOKE TEST PASSED")
print("  obs shapes:", shapes)
print("  reward:", round(float(rew), 4))
'@

    $tmpScript = Join-Path $env:TEMP "robomme_smoke_test.py"
    Set-Content $tmpScript $smokeScript -Encoding UTF8

    & $Python $tmpScript
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[FAIL] Smoke test failed. See error above and consult TRAINING_SETUP_WINDOWS.md troubleshooting." -ForegroundColor Red
        exit 1
    }
    Remove-Item $tmpScript -ErrorAction SilentlyContinue
}

# ── Done ───────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Setup complete! Quick-start commands:" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Baseline 1 (Vanilla PPO):"
Write-Host "    .venv\Scripts\python.exe -m train.train_ppo --task BinFill --timesteps 500000 --n_envs 2 --outdir runs/ppo"
Write-Host ""
Write-Host "  Baseline 2 (PPO + ICM curiosity):"
Write-Host "    .venv\Scripts\python.exe -m train.train_ppo_icm --task BinFill --timesteps 500000 --n_envs 2 --outdir runs/ppo_icm"
Write-Host ""
Write-Host "  Baseline 3 (PPO + PTP Memory):"
Write-Host "    .venv\Scripts\python.exe -m train.train_ppo_ptp --task BinFill --timesteps 500000 --n_envs 2 --outdir runs/ppo_ptp"
Write-Host ""
Write-Host "  TensorBoard:"
Write-Host "    .venv\Scripts\python.exe -m tensorboard.main --logdir runs/"
Write-Host ""
Write-Host "  Full setup guide: TRAINING_SETUP_WINDOWS.md"
Write-Host ""
