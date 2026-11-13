# Reproduce the uploads from the raw CSVs in upi-fraud-rings/ (README.md "Run").
# usage (repo root):  .\run_all.ps1 [-Tag repro] [-Threads 10] [-SkipPrep] [-H6Only]
# Outputs in work/subs/ (reference copies in brackets):
#   <Tag>_ppq.csv            [h6_ppq.csv]        h6 model + postproc --qr-null              (upload #1, public .5081)
#   ts_judge_<Tag>.csv       [ts_judge_h6.csv]   + device-id zone override + bait rule      (upload #3, public .5443)
#   <Tag>_final1.csv         [final1.csv]        c2s6 model + zones/bait + art_acct + art_zonemax (upload #4)
# -H6Only stops after <Tag>_ppq.csv. Module caches go to work/feats_cache (or $env:UPI_CACHE if set).
param([string]$Tag = "repro", [int]$Threads = 10, [switch]$SkipPrep, [switch]$H6Only)
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
if (-not $env:POLARS_MAX_THREADS) { $env:POLARS_MAX_THREADS = "8" }
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$cache = if ($env:UPI_CACHE) { $env:UPI_CACHE } else { "feats_cache" }
$mods = "flow,burst,takeover,scam,ring,pairs"
$c2 = "${Tag}_c2"
function TrainArgs([string]$tag, [string]$modules, [string]$matrix) {
    @("train.py", "--modules", $modules, "--matrix", $matrix, "--relabel", "C", "--rounds", "150", "--seeds", "6",
      "--threads", "$Threads", "--tag", $tag)
}
$train = TrainArgs $Tag $mods (Join-Path $cache "matrix_$Tag.parquet")
$train2 = TrainArgs $c2 "$mods,core2" (Join-Path $cache "matrix_$c2.parquet")

function Step([string]$name, [string[]]$cmd) {
    Write-Host "== $name" -ForegroundColor Cyan
    $t = Get-Date
    & $py @cmd
    if ($LASTEXITCODE -ne 0) { throw "$name failed (exit $LASTEXITCODE)" }
    Write-Host ("   {0:N0} s" -f ((Get-Date) - $t).TotalSeconds)
}

function Same([string]$out, [string]$ref) {  # byte comparison with the reference upload, if present
    if (-not (Test-Path "subs/$ref")) { return }
    $eq = (Get-FileHash "subs/$out").Hash -eq (Get-FileHash "subs/$ref").Hash
    Write-Host ("   subs/$out " + $(if ($eq) { "== subs/$ref (byte-identical)" } else { "DIFFERS from subs/$ref" })) -ForegroundColor $(if ($eq) { "Green" } else { "Red" })
}

Push-Location (Join-Path $PSScriptRoot "work")
try {
    if (-not $SkipPrep) { Step "1 prep: raw CSV -> data/*.parquet" @("prep.py") }
    Step "2-3 features + matrix" @("make_matrix.py", (Join-Path $cache "matrix_$Tag.parquet"))
    Step "4a train val (temporal CV, bias tuning)" ($train + @("--mode", "val"))
    Step "4b train full (all train rows -> test)" ($train + @("--mode", "full"))
    Step "5 post-processing" @("postproc.py", $Tag, "--qr-null")
    Same "${Tag}_ppq.csv" "h6_ppq.csv"
    if ($H6Only) { return }
    Step "6 device-id zones + bait rule" @("ts_judge_post.py", $Tag)
    Same "ts_judge_$Tag.csv" "ts_judge_h6.csv"
    Step "7a art_zonemax (mule/velocity pooled zone DP)" @("ts_judge_post.py", $Tag, "--mv", "--out", "art_zonemax_$Tag")
    Step "7b art_acct (velocity receiver propagation)" @("art_acct_post.py", $Tag, "art_acct_$Tag")
    Step "8 core2 matrix" @("make_matrix.py", (Join-Path $cache "matrix_$c2.parquet"), "--core2")
    Step "9a train val, core2" ($train2 + @("--mode", "val"))
    Step "9b train full, core2" ($train2 + @("--mode", "full"))
    Step "10 zones + bait on the core2 model" @("ts_judge_post.py", $c2)
    Step "11 transplant the two h6-based rules" @("combine.py", "ts_judge_$c2", "${Tag}_final1",
        "ts_judge_${Tag}:art_acct_$Tag", "ts_judge_${Tag}:art_zonemax_$Tag")
    Same "${Tag}_final1.csv" "final1.csv"
} finally { Pop-Location }
