#!/usr/bin/env bash
# 모든 예시 text input을 순차 실행해 outputs_v2/dit360/<name>/ 산출.
# 한 장면이 실패해도 다음 장면을 계속 진행.
#   ./run_all_scenes.sh            # 01~10 실행 (11_metal_factory는 이미 생성됨)
#   ./run_all_scenes.sh all        # 11 포함 전체 재생성
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

names=(01_cyberpunk 02_snow_village 03_ruined_temple 04_desert_night 05_coral_reef \
       06_bamboo_forest 07_mars_colony 08_medieval_village 09_tropical_beach 10_floating_island)
prompts=(
"A cyberpunk city at night, neon reflections on wet streets."
"A snowy mountain village at sunset, smoke from chimneys."
"An ancient ruined temple overgrown with vines in a jungle."
"A vast desert under a starry sky, dunes and the Milky Way."
"An underwater coral reef with colorful fish and a shipwreck."
"A misty bamboo forest at dawn with a quiet stone path."
"A Mars colony at golden hour, domes on rust-red terrain."
"A medieval autumn village with a stone cathedral in distance."
"A tropical beach at twilight with palm trees and pink clouds."
"A floating sky island with waterfalls and stone ruins."
)
if [ "${1:-}" = "all" ]; then
    names+=(11_metal_factory)
    prompts+=("A polished steel factory hangar with chrome pipes, brushed aluminum panels, and silver machinery under bright industrial lights.")
fi

mkdir -p "${HERE}/logs"
ok=0; fail=0
for i in "${!names[@]}"; do
    n="${names[$i]}"; p="${prompts[$i]}"
    echo "==================== [$((i+1))/${#names[@]}] ${n} ===================="
    if bash "${HERE}/run_hq_pipeline.sh" "${n}" "${p}" > "${HERE}/logs/${n}.log" 2>&1; then
        echo "  OK  -> outputs_v2/dit360/${n}/   (log: logs/${n}.log)"; ok=$((ok+1))
    else
        echo "  FAIL -> logs/${n}.log"; fail=$((fail+1))
    fi
done
echo "==================== DONE: ${ok} ok, ${fail} fail ===================="