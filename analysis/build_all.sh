#!/usr/bin/env bash
# Build every figure, table, prompt export, and example export that the paper
# (paper/main.tex plus the split appendices it cross-references) consumes, by
# running the analysis/plots and analysis/tables scripts via `uv run`. Outputs
# land in paper/figures/, paper/tables/, paper/prompts/, paper/examples/.
#
# Scripts in analysis/plots and analysis/tables are auto-discovered and run
# once with their defaults, except: (1) scripts that must run several times
# because the paper embeds several variants (e.g. ``--setting all --by-model``
# and ``--compare-settings``, or one ``--run`` per regression) are listed in
# ``SCRIPT_INVOCATIONS``, where each ``|``-separated entry is one invocation's
# extra args; (2) paper scripts living outside those two directories are
# listed in ``EXTRA_SCRIPTS``. Keep both in sync with paper/main.tex.
#
# Usage (from repo root: ./analysis/build_all.sh ; from analysis/: ./build_all.sh):
#   ./analysis/build_all.sh        # run all, abort on first failure
#   ./analysis/build_all.sh -k     # keep going past failing scripts
#   ./analysis/build_all.sh -n     # dry-run: list what would be executed
#   ./analysis/build_all.sh -h     # show this help
#
# Extra arguments after `--` are forwarded to every invocation, e.g.
#   ./analysis/build_all.sh -- --no-save   # print tables to stdout instead of saving

set -euo pipefail

# This file lives in analysis/; repo root is one level up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

KEEP_GOING=0
DRY_RUN=0
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -k|--keep-going) KEEP_GOING=1; shift ;;
        -n|--dry-run)    DRY_RUN=1;    shift ;;
        -h|--help)
            sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        --) shift; FORWARD_ARGS=("$@"); break ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Use -h for help." >&2
            exit 2 ;;
    esac
done

DIRS=(
    "analysis/plots"
    "analysis/tables"
)

# Per-script invocation matrices, keyed by script path relative to the repo
# root. Each value is a ``|``-separated list; the script is invoked once per
# entry, with the entry's contents word-split into argv. An empty entry means
# "run with no extra args" (i.e. defaults), so a script that needs both its
# default run and a second run with overrides should be listed as
# ``"|--flag value"``.
#
# Scripts not present in this map run once with no extra args.
declare -A SCRIPT_INVOCATIONS

# Paper scripts outside analysis/plots and analysis/tables. Run once each with
# defaults (or add a SCRIPT_INVOCATIONS entry).
declare -a EXTRA_SCRIPTS=(
    # Respondent demographic figures (*_distribution.pdf) for the human-baseline
    # appendix (paper/appendices/appendix_N.tex).
    "analysis/human_baseline/plots.py"
    # Posterior-predictive human-vs-LLM gap from the M1 posterior (figures +
    # paper/tables/regressions/human_ai_gap.tex); needs the fitted M1 run.
    "analysis/regressions/posterior_human_ai_gap.py"
)

# Paper example completions for two summary IDs.
SCRIPT_INVOCATIONS["analysis/plots/export_paper_examples.py"]=$'--summary_ids 526-1 1955-2'

# Two paper variants: all settings pooled and the cross-setting comparison
# figure with hatching for each task.
SCRIPT_INVOCATIONS["analysis/plots/plot_experimental_conditions.py"]=$'--setting all --by-model|--compare-settings'

# Aggregated condition-type plot variants: pooled, full cross-setting, and
# compact fraud/refusal-only cross-setting figure for the main text.
SCRIPT_INVOCATIONS["analysis/plots/plot_aggregated_conditions.py"]=$'--setting all|--compare-settings|--compare-settings-main-text'

# Same two paper variants for every conditional-rate family
# (P(A|F), P(F|A), P(R|A), P(A|R)).
SCRIPT_INVOCATIONS["analysis/plots/plot_proportion_aware.py"]=$'--setting all --by-model|--compare-settings'

# One regression table set per fitted run (paper/tables/regressions/*{,_m2,_m3}.tex
# for M1, M2 and M3 respectively).
SCRIPT_INVOCATIONS["analysis/tables/table_regression_coefs.py"]=$'--run m1|--run m2|--run m3'

declare -a SCRIPTS=()
for dir in "${DIRS[@]}"; do
    if [[ ! -d "$dir" ]]; then
        echo "warning: directory not found, skipping: $dir" >&2
        continue
    fi
    while IFS= read -r -d '' script; do
        script="${script#./}"
        SCRIPTS+=("$script")
    done < <(find "$dir" -type f -name '*.py' ! -path '*/__pycache__/*' ! -name '__*' -print0 | sort -z)
done
for script in "${EXTRA_SCRIPTS[@]}"; do
    if [[ ! -f "$script" ]]; then
        echo "warning: extra script not found, skipping: $script" >&2
        continue
    fi
    SCRIPTS+=("$script")
done

if [[ ${#SCRIPTS[@]} -eq 0 ]]; then
    echo "No scripts found in: ${DIRS[*]}" >&2
    exit 1
fi

# Expand SCRIPTS into a flat list of (script, extra_args_string) pairs to
# execute. We use two parallel arrays because bash assoc arrays can't hold
# duplicate keys, and a script can appear multiple times in SCRIPT_INVOCATIONS.
declare -a RUN_SCRIPTS=()
declare -a RUN_ARGS=()
for script in "${SCRIPTS[@]}"; do
    if [[ -n "${SCRIPT_INVOCATIONS[$script]+_}" ]]; then
        IFS='|' read -r -a invocations <<< "${SCRIPT_INVOCATIONS[$script]}"
        for inv in "${invocations[@]}"; do
            RUN_SCRIPTS+=("$script")
            RUN_ARGS+=("$inv")
        done
    else
        RUN_SCRIPTS+=("$script")
        RUN_ARGS+=("")
    fi
done

echo "Found ${#SCRIPTS[@]} script(s); ${#RUN_SCRIPTS[@]} total invocation(s):"
for i in "${!RUN_SCRIPTS[@]}"; do
    if [[ -n "${RUN_ARGS[$i]}" ]]; then
        echo "  - ${RUN_SCRIPTS[$i]} ${RUN_ARGS[$i]}"
    else
        echo "  - ${RUN_SCRIPTS[$i]}"
    fi
done

if [[ $DRY_RUN -eq 1 ]]; then
    exit 0
fi

declare -a OK=()
declare -a FAILED=()

for i in "${!RUN_SCRIPTS[@]}"; do
    script="${RUN_SCRIPTS[$i]}"
    args_str="${RUN_ARGS[$i]}"
    extra_args=()
    if [[ -n "$args_str" ]]; then
        # Word-split the args string intentionally (no quoting inside the map).
        eval "extra_args=($args_str)"
    fi

    label="$script ${args_str}"
    echo
    echo "=========================================="
    echo ">>> uv run $script ${args_str} ${FORWARD_ARGS[*]:-}"
    echo "=========================================="

    if uv run "$script" "${extra_args[@]}" "${FORWARD_ARGS[@]}"; then
        OK+=("$label")
    else
        rc=$?
        FAILED+=("$label (exit $rc)")
        if [[ $KEEP_GOING -eq 0 ]]; then
            echo
            echo "Aborting: $label failed with exit $rc" >&2
            echo "Use -k/--keep-going to continue past failures." >&2
            exit "$rc"
        fi
    fi
done

echo
echo "=========================================="
echo "Summary: ${#OK[@]} succeeded, ${#FAILED[@]} failed"
echo "=========================================="
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed scripts:"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
    exit 1
fi
