param([switch]$PreflightOnly)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$pythonExe = Join-Path $projectRoot 'crossres_pred/.venv/Scripts/python.exe'
$pythonWindowless = Join-Path $projectRoot 'crossres_pred/.venv/Scripts/pythonw.exe'
$runnerPath = Join-Path $PSScriptRoot 'evaluate_f0_variants.py'
$configPath = Join-Path $projectRoot 'crossres_pred/configs/f0_variants_20260908.json'
$receiptPath = Join-Path $projectRoot 'output/crossres_data/f0_family_20260908/variants/preflight_tests.json'
$taskName = 'VesuviusCrossres-F0Variants-20260908'

foreach ($inputPath in @($pythonExe, $pythonWindowless, $runnerPath, $configPath)) {
    if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) {
        throw "variants launch input missing: $inputPath"
    }
}
$configDigest = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash.ToLowerInvariant()

# CPU tests for the instrumentation and the ladder logic must pass for this exact config.
Push-Location (Join-Path $projectRoot 'crossres_pred')
try {
    & $pythonExe -m pytest -q -p no:cacheprovider `
        tests/test_audit_per_patch.py tests/test_audit_bootstrap.py `
        tests/test_checkpoint_average.py tests/test_f0_variants.py
    $testsPassed = ($LASTEXITCODE -eq 0)
} finally {
    Pop-Location
}
if (-not $testsPassed) { throw 'CPU tests failed; no task launched' }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $receiptPath) | Out-Null
@{ passed = $true; config_sha256 = $configDigest; at = (Get-Date).ToUniversalTime().ToString('o') } |
    ConvertTo-Json | Set-Content -LiteralPath $receiptPath -Encoding UTF8

& $pythonExe $runnerPath --config $configPath --config-sha256 $configDigest --preflight-only
if ($LASTEXITCODE -ne 0) { throw 'variants preflight failed; no task launched' }
if ($PreflightOnly) { exit 0 }

$taskArguments = '"' + $runnerPath + '" --config "' + $configPath + '" --config-sha256 ' + $configDigest
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($null -ne $existingTask) {
    if ($existingTask.Actions[0].Arguments -ne $taskArguments) {
        throw 'A variants task with a different config already exists; refusing to replace it'
    }
    if ([string]$existingTask.State -eq 'Running') {
        Write-Output 'The variants task is already running; no second instance created'
        exit 0
    }
}
$conflicts = @(Get-ScheduledTask -TaskName 'VesuviusCrossres-*' | Where-Object {
    $_.TaskName -ne $taskName -and ([string]$_.State -eq 'Running')
})
if ($conflicts.Count -gt 0) {
    throw ('Other crossres tasks are running: ' + ($conflicts.TaskName -join ', '))
}

if ($null -eq $existingTask) {
    $operatorIdentity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    $principal = New-ScheduledTaskPrincipal -UserId $operatorIdentity -LogonType Interactive -RunLevel Limited
    $action = New-ScheduledTaskAction -Execute $pythonWindowless -Argument $taskArguments -WorkingDirectory $projectRoot
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $operatorIdentity
    $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Days 14) `
        -Priority 6 -Hidden -StartWhenAvailable
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal `
        -Settings $settings -Description 'Pre-registered F0 inference-variant ladder on the frozen v14p2 harness; journalled; GPU-lock serialized; operator review pause' | Out-Null
} else {
    Enable-ScheduledTask -TaskName $taskName | Out-Null
}
Start-ScheduledTask -TaskName $taskName
Write-Output "Started task $taskName with config SHA256 $configDigest"
