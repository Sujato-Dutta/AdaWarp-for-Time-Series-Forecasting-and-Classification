#!/bin/bash
set -euo pipefail

REPO_ROOT="${ADAWARP_REPO_ROOT:-$WORK/motion_code-master}"
cd "$REPO_ROOT"
mkdir -p physics_modal_v4/enhancements/logs physics_modal_v4/enhancements/results

MODE="${1:-smoke}"
SHARDS="${LATENTFLOW_ENHANCEMENT_SHARDS:-16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/physics_modal_v4/enhancements/results}"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_ROOT/physics_modal_v4/results/final}"

submit_array() {
  local experiment="$1" groups="$2" walltime="$3" name="$4"
  local shards="$SHARDS"
  (( groups < shards )) && shards="$groups"
  sbatch --array="0-$((shards-1))%$shards" -t "$walltime" -J "$name" \
    --export="ALL,EXPERIMENT=$experiment,SHARD_COUNT=$shards,OUTPUT_ROOT=$OUTPUT_ROOT,SOURCE_ROOT=$SOURCE_ROOT" \
    physics_modal_v4/enhancements/vista.sbatch
}

extract_job_id() {
  local raw="$1" job_id
  job_id=$(printf '%s\n' "$raw" | awk -F';' '
    {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
      if ($1 ~ /^[0-9]+$/) id=$1
      if ($0 ~ /Submitted batch job [0-9]+/) {
        n=split($0, fields, /[[:space:]]+/)
        id=fields[n]
      }
    }
    END { print id }
  ')
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "Could not parse job ID from sbatch output:" >&2
    printf '%s\n' "$raw" >&2
    return 1
  }
  printf '%s\n' "$job_id"
}

submit_selector_repair() {
  local central_job factorial_job table_job raw
  central_job="${EXISTING_CENTRAL_JOB:-}"
  if [[ -z "$central_job" ]]; then
    central_job=$(
      squeue -h -u "$USER" -n lf_rcent -o "%A" 2>/dev/null \
        | awk '/^[0-9]+$/ {print}' | sort -n | tail -n 1
    )
  fi
  if [[ -n "$central_job" ]]; then
    echo "Reusing active central repair job $central_job" >&2
  else
    raw=$(sbatch --parsable --array="0-$((SHARDS-1))%$SHARDS" \
      -t 08:00:00 -J lf_rcent \
      --export="ALL,EXPERIMENT=selector_repair_central,SHARD_COUNT=$SHARDS,OUTPUT_ROOT=$OUTPUT_ROOT,SOURCE_ROOT=$SOURCE_ROOT" \
      physics_modal_v4/enhancements/vista.sbatch)
    central_job=$(extract_job_id "$raw")
    printf '%s\n' "$raw" >&2
  fi

  if ! raw=$(sbatch --parsable --dependency="afterok:$central_job" \
    --array="0-$((SHARDS-1))%$SHARDS" -t 08:00:00 -J lf_rfact \
    --export="ALL,EXPERIMENT=selector_repair_factorial,SHARD_COUNT=$SHARDS,OUTPUT_ROOT=$OUTPUT_ROOT,SOURCE_ROOT=$SOURCE_ROOT" \
    physics_modal_v4/enhancements/vista.sbatch 2>&1); then
    printf '%s\n' "$raw" >&2
    echo "Central repair remains submitted as job $central_job." >&2
    return 1
  fi
  factorial_job=$(extract_job_id "$raw")
  printf '%s\n' "$raw" >&2

  if ! raw=$(sbatch --parsable -p gh-dev --dependency="afterok:$factorial_job" \
    -t 00:30:00 -J lf_rtabs \
    --export="ALL,EXPERIMENT=selector_repair_tables,OUTPUT_ROOT=$OUTPUT_ROOT,SOURCE_ROOT=$SOURCE_ROOT" \
    physics_modal_v4/enhancements/vista.sbatch 2>&1); then
    printf '%s\n' "$raw" >&2
    echo "Central=$central_job and factorial=$factorial_job remain submitted." >&2
    return 1
  fi
  table_job=$(extract_job_id "$raw")
  printf '%s\n' "$raw" >&2
  echo "selector repair submitted central=$central_job factorial=$factorial_job tables=$table_job"
}

submit_remaining_all() {
  local raw smoke_job control_job curve_job aggregate_job
  raw=$(sbatch --parsable -p gh-dev -t 02:00:00 -J lf_remsm \
    --export="ALL,EXPERIMENT=remaining_smoke,OUTPUT_ROOT=$OUTPUT_ROOT,SOURCE_ROOT=$SOURCE_ROOT" \
    physics_modal_v4/enhancements/vista.sbatch)
  smoke_job=$(extract_job_id "$raw")
  printf '%s\n' "$raw" >&2

  raw=$(sbatch --parsable --dependency="afterok:$smoke_job" \
    --array="0-$((SHARDS-1))%$SHARDS" -t 2-00:00:00 -J lf_ngp5 \
    --export="ALL,EXPERIMENT=matched_controls_multiseed,SHARD_COUNT=$SHARDS,OUTPUT_ROOT=$OUTPUT_ROOT,SOURCE_ROOT=$SOURCE_ROOT" \
    physics_modal_v4/enhancements/vista.sbatch)
  control_job=$(extract_job_id "$raw")
  printf '%s\n' "$raw" >&2

  raw=$(sbatch --parsable --dependency="afterok:$smoke_job" \
    --array="0-$((SHARDS-1))%$SHARDS" -t 2-00:00:00 -J lf_tpcur \
    --export="ALL,EXPERIMENT=timepro_data_curve,SHARD_COUNT=$SHARDS,OUTPUT_ROOT=$OUTPUT_ROOT,SOURCE_ROOT=$SOURCE_ROOT" \
    physics_modal_v4/enhancements/vista.sbatch)
  curve_job=$(extract_job_id "$raw")
  printf '%s\n' "$raw" >&2

  raw=$(sbatch --parsable -p gh-dev \
    --dependency="afterok:$control_job:$curve_job" -t 00:30:00 -J lf_remagg \
    --export="ALL,EXPERIMENT=remaining_aggregate,OUTPUT_ROOT=$OUTPUT_ROOT,SOURCE_ROOT=$SOURCE_ROOT" \
    physics_modal_v4/enhancements/vista.sbatch)
  aggregate_job=$(extract_job_id "$raw")
  printf '%s\n' "$raw" >&2
  echo "remaining checks submitted smoke=$smoke_job controls=$control_job curve=$curve_job aggregate=$aggregate_job"
}

case "$MODE" in
  remaining_all) submit_remaining_all ;;
  remaining_smoke)
    sbatch -p gh-dev -t 02:00:00 -J lf_remsm \
      --export="ALL,EXPERIMENT=remaining_smoke,OUTPUT_ROOT=$OUTPUT_ROOT,SOURCE_ROOT=$SOURCE_ROOT" \
      physics_modal_v4/enhancements/vista.sbatch
    ;;
  submission_smoke)
    sbatch -p gh-dev -t 00:30:00 -J lf_subsmk \
      --export="ALL,EXPERIMENT=submission_smoke,OUTPUT_ROOT=$OUTPUT_ROOT,SOURCE_ROOT=$SOURCE_ROOT" \
      physics_modal_v4/enhancements/vista.sbatch
    ;;
  traffic_replay)
    sbatch -t 12:00:00 -J lf_trrep \
      --export="ALL,EXPERIMENT=traffic_replay,OUTPUT_ROOT=$OUTPUT_ROOT,SOURCE_ROOT=$SOURCE_ROOT" \
      physics_modal_v4/enhancements/vista.sbatch
    ;;
  submission_provenance) submit_array submission_provenance 140 08:00:00 lf_prov ;;
  submission_selector) submit_array submission_selector 140 2-00:00:00 lf_selau ;;
  submission_numerical) submit_array submission_numerical 28 08:00:00 lf_numau ;;
  submission_fusion_profile) submit_array submission_fusion_profile 28 08:00:00 lf_fprof ;;
  matched_controls) submit_array matched_controls 28 2-00:00:00 lf_ngp ;;
  matched_controls_multiseed) submit_array matched_controls_multiseed 112 2-00:00:00 lf_ngp5 ;;
  submission_aggregate)
    sbatch -p gh-dev -t 00:30:00 -J lf_subagg \
      --export="ALL,EXPERIMENT=submission_aggregate,OUTPUT_ROOT=$OUTPUT_ROOT,SOURCE_ROOT=$SOURCE_ROOT" \
      physics_modal_v4/enhancements/vista.sbatch
    ;;
  remaining_aggregate)
    sbatch -p gh-dev -t 00:30:00 -J lf_remagg \
      --export="ALL,EXPERIMENT=remaining_aggregate,OUTPUT_ROOT=$OUTPUT_ROOT,SOURCE_ROOT=$SOURCE_ROOT" \
      physics_modal_v4/enhancements/vista.sbatch
    ;;
  smoke)
    sbatch -p gh-dev -t 00:30:00 -J lf_enh_smoke \
      --export="ALL,EXPERIMENT=smoke,OUTPUT_ROOT=$OUTPUT_ROOT,SOURCE_ROOT=$SOURCE_ROOT" \
      physics_modal_v4/enhancements/vista.sbatch
    ;;
  central) submit_array central 140 2-00:00:00 lf_cent5 ;;
  factorial) submit_array factorial 84 2-00:00:00 lf_fact3 ;;
  repair_ablation_tables) submit_selector_repair ;;
  strongest) submit_array strongest 112 2-00:00:00 lf_strong ;;
  capacity) submit_array capacity 28 2-00:00:00 lf_cap ;;
  full_data) submit_array full_data 28 2-00:00:00 lf_fulld ;;
  data_curve) submit_array data_curve 112 2-00:00:00 lf_curve ;;
  timepro_data_curve) submit_array timepro_data_curve 84 2-00:00:00 lf_tpcur ;;
  channel_order) submit_array channel_order 4 12:00:00 lf_chord ;;
  channel_order_expanded) submit_array channel_order_expanded 6 12:00:00 lf_chx5 ;;
  diagnostics) submit_array diagnostics 7 04:00:00 lf_diag ;;
  synthetic) submit_array synthetic 35 2-00:00:00 lf_syn2 ;;
  real_scaling) submit_array real_scaling 6 2-00:00:00 lf_rscale ;;
  reviewer_smoke)
    sbatch -p gh-dev -t 00:30:00 -J lf_revsmk \
      --export="ALL,EXPERIMENT=reviewer_smoke,OUTPUT_ROOT=$OUTPUT_ROOT,SOURCE_ROOT=$SOURCE_ROOT" \
      physics_modal_v4/enhancements/vista.sbatch
    ;;
  prior_controls) submit_array prior_controls 252 2-00:00:00 lf_prior3 ;;
  checkpoint_audits) submit_array checkpoint_audits 140 2-00:00:00 lf_ckpaud ;;
  final_audit_analysis)
    sbatch -p gh-dev -t 00:30:00 -J lf_revan \
      --export="ALL,EXPERIMENT=final_audit_analysis,OUTPUT_ROOT=$OUTPUT_ROOT,SOURCE_ROOT=$SOURCE_ROOT" \
      physics_modal_v4/enhancements/vista.sbatch
    ;;
  aggregate)
    sbatch -p gh-dev -t 00:30:00 -J lf_enh_stats \
      --export="ALL,EXPERIMENT=aggregate,OUTPUT_ROOT=$OUTPUT_ROOT,SOURCE_ROOT=$SOURCE_ROOT" \
      physics_modal_v4/enhancements/vista.sbatch
    ;;
  *)
    echo "usage: $0 {remaining_all|remaining_smoke|submission_smoke|traffic_replay|submission_provenance|submission_selector|submission_numerical|submission_fusion_profile|matched_controls|matched_controls_multiseed|submission_aggregate|remaining_aggregate|smoke|central|factorial|repair_ablation_tables|strongest|capacity|full_data|data_curve|timepro_data_curve|diagnostics|channel_order|channel_order_expanded|synthetic|real_scaling|reviewer_smoke|prior_controls|checkpoint_audits|final_audit_analysis|aggregate}" >&2
    exit 2
    ;;
esac
