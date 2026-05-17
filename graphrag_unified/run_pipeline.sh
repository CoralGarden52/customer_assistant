#!/bin/bash
# Complete pipeline for GraphRAG multimodal submission
# Usage: bash run_pipeline.sh [--clean]
#   --clean: Clear cache and output before starting (fresh start)

set -e  # Exit on error

PYTHON="/home/wyz/miniconda3/envs/wyz/bin/python3"
GRAPHRAG="/home/wyz/miniconda3/envs/wyz/bin/graphrag"
PROJECT_DIR="/home/wyz/kefu/graphrag_unified"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/pipeline_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

# Log function
log() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Clean cache and output if --clean flag is passed
if [[ "$1" == "--clean" ]]; then
    log "Cleaning cache and output directories..."
    rm -rf "$PROJECT_DIR/cache/"*
    rm -rf "$PROJECT_DIR/output/"*
    rm -f "$PROJECT_DIR/submission.csv"
    log "Done."
fi

log "=========================================="
log "GraphRAG Multimodal Pipeline"
log "Started: $(date)"
log "Log file: $LOG_FILE"
log "=========================================="

# Step 1: Preprocess manuals (preserve <PIC> markers)
log ""
log "[1/4] Preprocessing manuals..."
$PYTHON preprocess_multimodal.py 2>&1 | tee -a "$LOG_FILE"

# Step 2: Build GraphRAG index
log ""
log "[2/4] Building GraphRAG index..."
$GRAPHRAG index 2>&1 | tee -a "$LOG_FILE"

# Step 3: Generate submission
log ""
log "[3/4] Generating submission..."
$PYTHON submit.py 2>&1 | tee -a "$LOG_FILE"

# Step 4: Sort submission by ID
log ""
log "[4/4] Sorting submission..."
$PYTHON sort_submission.py 2>&1 | tee -a "$LOG_FILE"

log ""
log "=========================================="
log "Pipeline complete!"
log "Finished: $(date)"
log "Output: $PROJECT_DIR/submission.csv"
log "Log: $LOG_FILE"
log "=========================================="
