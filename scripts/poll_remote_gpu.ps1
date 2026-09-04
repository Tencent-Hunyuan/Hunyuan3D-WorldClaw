<#
.SYNOPSIS
  Poll a remote NVIDIA host and show a Windows popup when a GPU is available.

.EXAMPLE
  .\scripts\poll_remote_gpu.ps1

.EXAMPLE
  .\scripts\poll_remote_gpu.ps1 -HostName ziplab-5090 -FreeMemoryGiB 28 -IntervalSeconds 60

.EXAMPLE
  .\scripts\poll_remote_gpu.ps1 -Once
#>
[CmdletBinding()]
param(
    [string]$HostName = "ziplab-5090",
    [double]$FreeMemoryGiB = 28.0,
    [int]$IntervalSeconds = 60,
    [double]$ReserveMarginGiB = 0.5,
    [string]$RemotePython = "/mnt/data/v-huguangyu/worldclaw-oss/envs/hunyuan3d/bin/python",
    [switch]$Once
)

$ErrorActionPreference = "Stop"
if ($IntervalSeconds -lt 5) {
    throw "IntervalSeconds must be at least 5."
}
if ($FreeMemoryGiB -le 0) {
    throw "FreeMemoryGiB must be positive."
}
if ($ReserveMarginGiB -lt 0) {
    throw "ReserveMarginGiB cannot be negative."
}

function Show-GpuAlert {
    param([string]$Message)

    try {
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.MessageBox]::Show(
            $Message,
            "Remote GPU available",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Information
        ) | Out-Null
    }
    catch {
        # Keep the monitor useful in PowerShell Core sessions without WinForms.
        [Console]::Beep(1000, 700)
        Write-Warning $Message
    }
}

function Get-RemoteGpuStatus {
    $query = 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'
    $lines = @(ssh $HostName $query 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw ("SSH/nvidia-smi failed on {0}: {1}" -f $HostName, ($lines -join " "))
    }

    foreach ($line in $lines) {
        if ($line -notmatch '^\s*(\d+)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*$') {
            continue
        }
        $used = [double]$Matches[2]
        $total = [double]$Matches[3]
        [pscustomobject]@{
            Index = [int]$Matches[1]
            UsedMiB = $used
            TotalMiB = $total
            FreeMiB = $total - $used
            Utilization = [double]$Matches[4]
        }
    }
}

function Start-GpuReservation {
    param([int]$GpuIndex, [double]$AvailableMiB)

    $reserveGiB = [math]::Max(0.25, ($AvailableMiB / 1024.0) - $ReserveMarginGiB)
    $elements = [long][math]::Floor($reserveGiB * 1024 * 1024 * 1024 / 4)
    $code = "import time,torch; elements=$elements; tensor=torch.empty((elements,),dtype=torch.float32,device='cuda').fill_(0); print('worldclaw_gpu_lock GiB=$([math]::Round($reserveGiB,2))',flush=True); time.sleep(86400)"
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($code))
    $logPath = "/tmp/worldclaw-gpu-lock-gpu$GpuIndex.log"
    $remoteCommand = "echo $encoded | base64 -d | nohup env CUDA_VISIBLE_DEVICES=$GpuIndex $RemotePython - > $logPath 2>&1 & echo `$!"
    $output = @(ssh $HostName $remoteCommand 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw ("GPU lock launch failed: " + ($output -join " "))
    }
    $pidText = ($output | Where-Object { $_ -match '^\s*\d+\s*$' } | Select-Object -Last 1).ToString().Trim()
    if (-not $pidText) {
        throw ("GPU lock launched without a PID: " + ($output -join " "))
    }
    return [pscustomobject]@{ Pid = $pidText; ReservedGiB = $reserveGiB }
}

$alertedSignature = $null
do {
    try {
        $gpus = @(Get-RemoteGpuStatus)
        $available = @($gpus | Where-Object {
            ($_.FreeMiB / 1024.0) -ge $FreeMemoryGiB -and $_.Utilization -le 10
        })
        $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        if ($available.Count -gt 0) {
            $signature = (($available | ForEach-Object { $_.Index }) -join ",")
            $summary = ($available | ForEach-Object {
                "GPU {0}: {1:N1} GiB free, {2:N0}% util" -f $_.Index, ($_.FreeMiB / 1024.0), $_.Utilization
            }) -join "; "
            Write-Host "[$stamp] Available: $summary" -ForegroundColor Green
            if ($signature -ne $alertedSignature) {
                try {
                    $locks = @($available | ForEach-Object {
                        $lock = Start-GpuReservation -GpuIndex $_.Index -AvailableMiB $_.FreeMiB
                        "GPU $($_.Index): PID $($lock.Pid), reserved $([math]::Round($lock.ReservedGiB, 2)) GiB"
                    })
                    $lockSummary = $locks -join "; "
                    Show-GpuAlert ("{0}`n`n{1}`n`n{2}" -f $HostName, $summary, $lockSummary)
                    Write-Host "[$stamp] $lockSummary" -ForegroundColor Yellow
                }
                catch {
                    Show-GpuAlert ("{0}`n`n{1}`n`nGPU lock failed: {2}" -f $HostName, $summary, $_.Exception.Message)
                }
                $alertedSignature = $signature
            }
        }
        else {
            $alertedSignature = $null
            $summary = ($gpus | ForEach-Object {
                "GPU {0}: {1:N1}/{2:N1} GiB used, {3:N0}% util" -f $_.Index, ($_.UsedMiB / 1024.0), ($_.TotalMiB / 1024.0), $_.Utilization
            }) -join "; "
            Write-Host "[$stamp] No GPU meets ${FreeMemoryGiB} GiB free and <=10% util. $summary"
        }
    }
    catch {
        Write-Warning ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $_.Exception.Message)
    }

    if (-not $Once) {
        Start-Sleep -Seconds $IntervalSeconds
    }
} while (-not $Once)
