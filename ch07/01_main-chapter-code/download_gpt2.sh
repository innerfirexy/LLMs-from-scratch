#!/usr/bin/env bash
# Download GPT-2 weights files to gpt2/<model_size>/ without loading them into memory.
# Usage: ./download_gpt2.sh [model_size]
# Allowed sizes: 124M (default), 355M, 774M, 1558M

set -euo pipefail

MODEL_SIZE="${1:-124M}"
ALLOWED_SIZES=("124M" "355M" "774M" "1558M")

if [[ ! " ${ALLOWED_SIZES[*]} " =~ " ${MODEL_SIZE} " ]]; then
    echo "Error: Invalid model size '${MODEL_SIZE}'. Allowed sizes: ${ALLOWED_SIZES[*]}"
    exit 1
fi

MODEL_DIR="gpt2/${MODEL_SIZE}"
BASE_URL="https://openaipublic.blob.core.windows.net/gpt-2/models"
BACKUP_URL="https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"

FILENAMES=(
    "checkpoint"
    "encoder.json"
    "hparams.json"
    "model.ckpt.data-00000-of-00001"
    "model.ckpt.index"
    "model.ckpt.meta"
    "vocab.bpe"
)

# Determine downloader
if command -v aria2c &> /dev/null; then
    DOWNLOADER="aria2c"
elif command -v wget &> /dev/null; then
    DOWNLOADER="wget"
else
    echo "Error: Neither aria2c nor wget is installed. Please install one of them."
    exit 1
fi

mkdir -p "${MODEL_DIR}"

download_file() {
    local url="$1"
    local backup="$2"
    local dest="$3"
    local filename
    filename=$(basename "${dest}")

    if [[ -f "${dest}" ]]; then
        echo "File already exists: ${dest}"
        return 0
    fi

    echo "Downloading ${filename} ..."

    if [[ "${DOWNLOADER}" == "aria2c" ]]; then
        if aria2c --dir="$(dirname "${dest}")" --out="${filename}" --max-tries=3 "${url}"; then
            return 0
        elif [[ -n "${backup}" ]]; then
            echo "Primary URL failed. Trying backup URL ..."
            aria2c --dir="$(dirname "${dest}")" --out="${filename}" --max-tries=3 "${backup}"
            return 0
        else
            return 1
        fi
    else
        if wget -q --show-progress -O "${dest}" "${url}"; then
            return 0
        elif [[ -n "${backup}" ]]; then
            echo "Primary URL failed. Trying backup URL ..."
            wget -q --show-progress -O "${dest}" "${backup}"
            return 0
        else
            return 1
        fi
    fi
}

for filename in "${FILENAMES[@]}"; do
    FILE_URL="${BASE_URL}/${MODEL_SIZE}/${filename}"
    BACKUP_FILE_URL="${BACKUP_URL}/${MODEL_SIZE}/${filename}"
    FILE_PATH="${MODEL_DIR}/${filename}"

    if ! download_file "${FILE_URL}" "${BACKUP_FILE_URL}" "${FILE_PATH}"; then
        echo "Error: Failed to download ${filename} from both primary and backup URLs."
        exit 1
    fi
done

echo "All GPT-2 (${MODEL_SIZE}) files downloaded to ${MODEL_DIR}/"
