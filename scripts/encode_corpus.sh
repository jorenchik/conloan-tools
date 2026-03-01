#!/bin/bash

# 1. Environment and Parameter Validation
if [ -z "$CORPUS_REGISTRY" ]; then
  echo "Error: CORPUS_REGISTRY environment variable is not set."
  exit 1
fi

VRT_FILE="$1"
CORPUS_ID="$2"

if [ -z "$VRT_FILE" ] || [ -z "$CORPUS_ID" ]; then
  echo "Usage: $0 <input.vrt> <corpus_id>"
  exit 1
fi

# Normalize ID to lowercase
CORPUS_ID=$(echo "$CORPUS_ID" | tr '[:upper:]' '[:lower:]')

# 2. Path Configuration (Co-located with Registry)
# We place data in a 'data' folder sitting next to the 'registry' folder
CWB_BASE_DIR=$(dirname "$CORPUS_REGISTRY")
DATA_DIR="$CWB_BASE_DIR/data/$CORPUS_ID"
REGISTRY_FILE="$CORPUS_REGISTRY/$CORPUS_ID"

# 3. Cleanup Logic
cleanup() {
  if [ -f "$REGISTRY_FILE" ] || [ -d "$DATA_DIR" ]; then
    echo "Found existing data for $CORPUS_ID."
    read -p "Delete and overwrite? [y/N] " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
      rm -f "$REGISTRY_FILE"
      rm -rf "$DATA_DIR"
    else
      echo "Aborting."
      exit 1
    fi
  fi
  mkdir -p "$DATA_DIR"
}

cleanup

# 4. Encoding
echo "Encoding $CORPUS_ID to $DATA_DIR..."

cwb-encode -d "$DATA_DIR" -f "$VRT_FILE" -R "$REGISTRY_FILE" \
  -c utf8 -x \
  -P pos -P lemma \
  -S doc:0+id+reference+section \
  -S p:0 -S s:0 -S g:0

# cwb-encode -d "$DATA_DIR" -f "$VRT_FILE" -R "$REGISTRY_FILE" \
#   -c utf8 \
#   -P pos -P lemma \
#   -S doc:0+id+reference+section \
#   -S p:0 -S s:0 -S g:0 \
#   -v

# cwb-encode -d "$DATA_DIR" -f "$VRT_FILE" -R "$REGISTRY_FILE" \
#   -c utf8 \
#   -P pos -P lemma \
#   -S doc:0+title+source+author+authorgender+published+genre+keywords+fileref \
#   -S p:0 -S s:0 \
#   -v

if [ $? -eq 0 ]; then
  echo "Building indices..."
  
  # 5. Indexing
  cwb-makeall -V "${CORPUS_ID^^}"
  
  # Final status check
  echo "---------------------------------------------------"
  echo "Registry: $REGISTRY_FILE"
  echo "Data:     $DATA_DIR"
  echo "Corpus ${CORPUS_ID^^} is ready for CQP."
else
  echo "Error: Encoding failed."
  exit 1
fi
