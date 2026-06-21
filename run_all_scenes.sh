#!/usr/bin/env bash
# 문자열 여러개 모드: 아래 목록(이름|프롬프트)을 편집한 뒤 실행하면 각각 산출.
# 사용: ./run_all_scenes.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# ===== 여기 편집: "이름|프롬프트" =====
items=(
"sunny_plaza|A sunny European plaza at noon, cobblestone ground, stone fountain, clear blue sky."
"cozy_library|A cozy wooden library, leather armchairs, brass lamps, tall bookshelves, warm light."
"cyberpunk|A cyberpunk city at night, neon reflections on wet streets."
"snow_village|A snowy mountain village at sunset, smoke from chimneys."
"ruined_temple|An ancient ruined temple overgrown with vines in a jungle."
"desert_night|A vast desert under a starry sky, dunes and the Milky Way."
"coral_reef|An underwater coral reef with colorful fish and a shipwreck."
"bamboo_forest|A misty bamboo forest at dawn with a quiet stone path."
"mars_colony|A Mars colony at golden hour, domes on rust-red terrain."
"medieval_village|A medieval autumn village with a stone cathedral in distance."
"tropical_beach|A tropical beach at twilight with palm trees and pink clouds."
"floating_island|A floating sky island with waterfalls and stone ruins."
"metal_factory|A polished steel factory hangar with chrome pipes, brushed aluminum panels, and silver machinery under bright industrial lights."
)
# =====================================

for it in "${items[@]}"; do
    name="${it%%|*}"; prompt="${it#*|}"
    echo "########## ${name} ##########"
    bash "${HERE}/run_hq_pipeline.sh" "${name}" "${prompt}" || echo "  (실패: ${name})"
done
