param(
    [int]$WaitProcessId,
    [string]$MinimaxLogPath,
    [string]$MinimaxScorePath = "evaluation_pipeline\v5\workspaces\allconv_v5_minimax\score_summary_answerhistory.json",
    [string]$GeminiCommand = "python v5/run.py --model-id gemini",
    [string]$MinimaxResumeCommand = "python v5/run.py --model-id minimax --from search",
    [string]$FallbackMinimaxApiKey = "",
    [string]$SecretsPath = "evaluation_pipeline\configs\matrix_secrets.yaml",
    [bool]$AutoLaunchGemini = $true
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$pipelineRoot = Join-Path $repoRoot "evaluation_pipeline"
$logsDir = Join-Path $PSScriptRoot "workspaces\logs"
New-Item -ItemType Directory -Force $logsDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$chainLog = Join-Path $logsDir "chain_minimax_to_gemini_$stamp.log"
$resolvedScorePath = if ([System.IO.Path]::IsPathRooted($MinimaxScorePath)) { $MinimaxScorePath } else { Join-Path $repoRoot $MinimaxScorePath }
$resolvedSecretsPath = if ([System.IO.Path]::IsPathRooted($SecretsPath)) { $SecretsPath } else { Join-Path $repoRoot $SecretsPath }
$runLogPointer = Join-Path $logsDir "run_log_current.txt"

function Write-ChainLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $chainLog -Value $line -Encoding UTF8
}

function Wait-ForProcessExit {
    param([int]$ProcessId)
    while ($true) {
        $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if (-not $proc) {
            return
        }
        Start-Sleep -Seconds 30
    }
}

function Test-MinimaxCompleted {
    param(
        [string]$LogPath,
        [string]$ScorePath
    )
    if (Test-Path $ScorePath) {
        Write-ChainLog "Detected score file: $ScorePath"
        return $true
    }
    if ((Test-Path $LogPath) -and (Select-String -Path $LogPath -Pattern "\[pipeline\] all steps completed\." -Quiet)) {
        Write-ChainLog "Detected completion marker in minimax log."
        return $true
    }
    return $false
}

function Test-BalanceFailure {
    param([string]$LogPath)
    if (-not (Test-Path $LogPath)) {
        return $false
    }
    $patterns = @(
        "402",
        "insufficient",
        "balance",
        "quota",
        "Payment Required",
        "credit"
    )
    foreach ($pattern in $patterns) {
        if (Select-String -Path $LogPath -Pattern $pattern -Quiet) {
            Write-ChainLog "Detected possible balance/quota pattern '$pattern' in log."
            return $true
        }
    }
    return $false
}

function Set-MinimaxKey {
    param(
        [string]$YamlPath,
        [string]$ApiKey
    )
    $content = Get-Content $YamlPath -Raw -Encoding UTF8
    $updated = [System.Text.RegularExpressions.Regex]::Replace(
        $content,
        '(?ms)(minimax:\s*\r?\n\s*api_key:\s*")[^"]+(")',
        ('$1' + [System.Text.RegularExpressions.Regex]::Escape($ApiKey).Replace('\', '\') + '$2')
    )
    if ($updated -eq $content) {
        throw "failed to update minimax api_key in $YamlPath"
    }
    Set-Content -Path $YamlPath -Value $updated -Encoding UTF8
    Write-ChainLog "Switched minimax api_key in $YamlPath"
}

function Start-ManagedPython {
    param(
        [string[]]$Arguments,
        [string]$OutPath,
        [string]$ErrPath
    )
    return Start-Process -FilePath python `
        -ArgumentList $Arguments `
        -WorkingDirectory $pipelineRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $OutPath `
        -RedirectStandardError $ErrPath `
        -PassThru
}

function Resolve-CurrentRunLog {
    if (Test-Path $runLogPointer) {
        return (Get-Content $runLogPointer -Raw -Encoding UTF8).Trim()
    }
    return ""
}

$currentPid = $WaitProcessId
$currentLog = $MinimaxLogPath
$fallbackUsed = $false

Write-ChainLog "Watcher started. wait_pid=$WaitProcessId minimax_log=$MinimaxLogPath"

while ($true) {
    Wait-ForProcessExit -ProcessId $currentPid
    Write-ChainLog "Process exited. pid=$currentPid"

    if (Test-MinimaxCompleted -LogPath $currentLog -ScorePath $resolvedScorePath) {
        break
    }

    if ((-not $fallbackUsed) -and $FallbackMinimaxApiKey -and (Test-BalanceFailure -LogPath $currentLog)) {
        Write-ChainLog "Minimax appears to have failed due to balance/quota. Switching back to fallback key and resuming from search."
        Set-MinimaxKey -YamlPath $resolvedSecretsPath -ApiKey $FallbackMinimaxApiKey

        $retryStamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $retryOut = Join-Path $logsDir "relaunch_minimax_$retryStamp.out.txt"
        $retryErr = Join-Path $logsDir "relaunch_minimax_$retryStamp.err.txt"
        $retryProc = Start-ManagedPython -Arguments @("v5/run.py", "--model-id", "minimax", "--from", "search") -OutPath $retryOut -ErrPath $retryErr
        Write-ChainLog "Relaunched minimax with fallback key. pid=$($retryProc.Id) stdout=$retryOut stderr=$retryErr"
        Start-Sleep -Seconds 3
        $resolvedRunLog = Resolve-CurrentRunLog
        if ($resolvedRunLog) {
            $currentLog = $resolvedRunLog
            Write-ChainLog "Updated current minimax log to $currentLog"
        }
        $currentPid = $retryProc.Id
        $fallbackUsed = $true
        continue
    }

    Write-ChainLog "Minimax run did not finish successfully. Gemini will not auto-start."
    exit 1
}

if (-not $AutoLaunchGemini) {
    Write-ChainLog "Minimax completed. AutoLaunchGemini=false, watcher exits without launching gemini."
    exit 0
}

Write-ChainLog "Launching gemini: $GeminiCommand"

$launchOut = Join-Path $logsDir "launch_gemini_$stamp.out.txt"
$launchErr = Join-Path $logsDir "launch_gemini_$stamp.err.txt"
$geminiProc = Start-ManagedPython -Arguments @("v5/run.py", "--model-id", "gemini") -OutPath $launchOut -ErrPath $launchErr

Write-ChainLog "Gemini launched. pid=$($geminiProc.Id) stdout=$launchOut stderr=$launchErr"
