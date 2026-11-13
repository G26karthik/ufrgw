# Reproduce the uploads from the raw CSVs in upi-fraud-rings/ (README.md "Run").
# usage (repo root):  .\run_all.ps1 [-Tag repro] [-Threads 10] [-SkipPrep] [-H6Only]
# Outputs in work/subs/ (reference copies in brackets):
#   <Tag>_ppq.csv            [h6_ppq.csv]        h6 model + postproc --qr-null              (upload #1, public .5081)
#   ts_judge_<Tag>.csv       [ts_judge_h6.csv]   + device-id zone override + bait rule      (upload #3, public .5443)
#   <Tag>_final1.csv         [final1.csv]        c2s6 model + zones/bait + art_acct + art_zonemax (upload #4)
#   <Tag>_v3_final / _sc_final / _bc_nr / _bc_bandv   rule layers on ts_judge_<Tag> (uploads #6, #7, #8, #9)
#   probe_edgefix_<Tag>_bc_bandv.csv  [probe_edgefix_bc_bandv.csv]  + 3 zone-edge rows (upload #10, public .5500)
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
    Same "art_zonemax_$Tag.csv" "art_zonemax.csv"
    Step "7b art_acct (velocity receiver propagation)" @("art_acct_post.py", $Tag, "art_acct_$Tag")
    Same "art_acct_$Tag.csv" "art_acct.csv"
    Step "8 core2 matrix" @("make_matrix.py", (Join-Path $cache "matrix_$c2.parquet"), "--core2")
    Step "9a train val, core2" ($train2 + @("--mode", "val"))
    Step "9b train full, core2" ($train2 + @("--mode", "full"))
    Step "10 zones + bait on the core2 model" @("ts_judge_post.py", $c2)
    Same "ts_judge_$c2.csv" "ts_judge_c2s6.csv"
    Step "11 transplant the two h6-based rules" @("combine.py", "ts_judge_$c2", "${Tag}_final1",
        "ts_judge_${Tag}:art_acct_$Tag", "ts_judge_${Tag}:art_zonemax_$Tag")
    Same "${Tag}_final1.csv" "final1.csv"
    Step "12 v3 rules (QRN merchant-QR, pair-restricted zone propagation)" @("v3_rules.py", "subs/ts_judge_$Tag.csv", "${Tag}_v3_final")
    Same "${Tag}_v3_final.csv" "v3_final.csv"
    Step "13 class-sweep rules (RS segment refund, FCB ph|qr edge)" @("sc_final.py", "subs/${Tag}_v3_final.csv", "${Tag}_sc_final")
    Same "${Tag}_sc_final.csv" "sc_final.csv"
    Step "14 big-class rules, no-regret (CZ chain mule, LFM fan-in mule, X999)" @("bc_final.py", "subs/${Tag}_sc_final.csv", "${Tag}_bc_nr", "CZ,LFM,X999")
    Same "${Tag}_bc_nr.csv" "bc_nr.csv"
    Step "15 BANDV band bet (DV0473782 fan-in -> velocity)" @("bc_final.py", "subs/${Tag}_bc_nr.csv", "${Tag}_bc_bandv", "BANDV")
    Same "${Tag}_bc_bandv.csv" "bc_bandv.csv"
    Step "16 EDGEFIX (3 zone-edge rows)" @("probe_rules.py", "EDGEFIX", "subs/${Tag}_bc_bandv.csv")
    Same "probe_edgefix_${Tag}_bc_bandv.csv" "probe_edgefix_bc_bandv.csv"
} finally { Pop-Location }
