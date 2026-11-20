# ==================================================================
#   get-screener.ps1
#   One command to install OR update the Stock Screener, then print
#   exactly where it lives.
#
#   Run it in PowerShell:
#     irm https://raw.githubusercontent.com/InventionConventionSpinOff/stock-screener/master/get-screener.ps1 | iex
#
#   - Not installed yet  -> downloads it to  %USERPROFILE%\stock-screener
#   - Already installed  -> updates the code in place and tells you
#                           the folder path
#   Your own data (inputs\, outputs\, cache\, logs\) is never touched.
#
#   Optional: set $env:STOCK_SCREENER_DIR before running to install
#   somewhere other than the default folder.
# ==================================================================
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$repoUrl = 'https://github.com/InventionConventionSpinOff/stock-screener'
$zipUrl  = "$repoUrl/archive/refs/heads/master.zip"

$target = $env:STOCK_SCREENER_DIR
if (-not $target) { $target = Join-Path $env:USERPROFILE 'stock-screener' }

$alreadyInstalled = Test-Path (Join-Path $target 'run_screener.py')
$haveGit = [bool](Get-Command git -ErrorAction SilentlyContinue)
$isGitCheckout = Test-Path (Join-Path $target '.git')
$userDataDirs = @('inputs', 'outputs', 'cache', 'logs')

function Update-FromZip {
    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("stock-screener-" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmp | Out-Null
    try {
        $zip = Join-Path $tmp 'repo.zip'
        Invoke-WebRequest -Uri $zipUrl -OutFile $zip -UseBasicParsing
        Expand-Archive -Path $zip -DestinationPath $tmp -Force
        $src = Get-ChildItem $tmp -Directory | Select-Object -First 1
        if (-not $src) { throw 'Downloaded archive had an unexpected layout.' }
        New-Item -ItemType Directory -Path $target -Force | Out-Null
        $roboArgs = @($src.FullName, $target, '/E', '/NJH', '/NJS', '/NDL', '/NP', '/NFL')
        if ($alreadyInstalled) { $roboArgs += '/XD'; $roboArgs += $userDataDirs }  # keep the user's own data on update
        & robocopy @roboArgs | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "File copy failed (robocopy exit code $LASTEXITCODE)." }
    }
    finally {
        Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if ($alreadyInstalled) {
    Write-Host "Stock Screener is already installed - updating it..."
} else {
    Write-Host "Installing Stock Screener..."
}

if ($isGitCheckout -and $haveGit) {
    $out = & git -C $target pull --ff-only 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host $out
        throw "Could not update (you may have edited tracked files). Folder: $target"
    }
    Write-Host ($out | Out-String).Trim()
}
elseif (-not $alreadyInstalled -and $haveGit -and -not (Test-Path $target)) {
    & git clone --quiet $repoUrl $target
    if ($LASTEXITCODE -ne 0) { throw 'git clone failed.' }
}
else {
    Update-FromZip
}

Write-Host ""
if ($alreadyInstalled) { Write-Host "Updated to the latest version." } else { Write-Host "Installed." }
Write-Host "Folder: $target"
Write-Host "Next: put your Positions CSV in '$target\inputs', then double-click RUN_SCREENER.bat"
