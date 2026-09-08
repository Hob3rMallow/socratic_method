param([switch]$PreflightOnly)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$pythonExe = Join-Path $projectRoot 'crossres_pred/.venv/Scripts/python.exe'
$pythonWindowless = Join-Path $projectRoot 'crossres_pred/.venv/Scripts/pythonw.exe'
$runnerPath = Join-Path $PSScriptRoot 'run_final_c3_250k.py'
$configPath = Join-Path $projectRoot 'crossres_pred/configs/final_c3_250k_20260905.json'
$receiptPath = Join-Path $projectRoot 'output/crossres_data/final_c3_250k_20260905/preflight_tests.json'
$taskName = 'VesuviusCrossres-FinalC3-250k'
$legacyTaskName = 'VesuviusCrossres-UnifiedLadder'

foreach ($inputPath in @($pythonExe, $pythonWindowless, $runnerPath, $configPath, $receiptPath)) {
    if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) {
        throw "F0 launch input missing: $inputPath"
    }
}
$configDigest = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash.ToLowerInvariant()
$testReceipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
if ($testReceipt.passed -ne $true -or $testReceipt.config_sha256 -ne $configDigest) {
    throw 'The passing CPU/preflight test receipt does not match this exact config'
}

& $pythonExe $runnerPath --config $configPath --config-sha256 $configDigest --preflight-only
if ($LASTEXITCODE -ne 0) { throw 'F0 sealed preflight failed; no task launched' }
if ($PreflightOnly) { exit 0 }

$taskArguments = '"' + $runnerPath + '" --config "' + $configPath + '" --config-sha256 ' + $configDigest
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($null -ne $existingTask) {
    if ($existingTask.Actions[0].Arguments -ne $taskArguments) {
        throw 'An F0 task with a different sealed config already exists; refusing to replace it'
    }
    if ([string]$existingTask.State -eq 'Running') {
        Write-Output 'The sealed F0 task is already running; no second instance created'
        exit 0
    }
}

$legacyTask = Get-ScheduledTask -TaskName $legacyTaskName -ErrorAction SilentlyContinue
if ($null -ne $legacyTask) {
    if ([string]$legacyTask.State -eq 'Running') {
        throw 'The historical ladder is running; leave it alone and review ownership'
    }
    Disable-ScheduledTask -TaskName $legacyTaskName | Out-Null
}
$conflicts = @(Get-ScheduledTask -TaskName 'VesuviusCrossres-*' | Where-Object {
    $_.TaskName -ne $taskName -and $_.Settings.Enabled
})
if ($conflicts.Count -gt 0) {
    throw ('Other crossres tasks remain enabled: ' + ($conflicts.TaskName -join ', '))
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
        -Settings $settings -Description 'One fresh-M7 C3 SGD/poly 250k F0, no trust ball; natural final commit; all-milestone fixed-.35 evaluation; operator review; 30-minute status log' | Out-Null
} else {
    Enable-ScheduledTask -TaskName $taskName | Out-Null
}
Start-ScheduledTask -TaskName $taskName
Write-Output "Started task $taskName with config SHA256 $configDigest"
Write-Output 'Only F0 is authorized. Existing evaluation workers are waited for, never killed. Legacy UnifiedLadder logon trigger is disabled.'
