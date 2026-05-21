#!/bin/bash

CONFIGS=("electricity" "energy" "etth" "ettm" "exchange_rate" "fmri" "illness" "stock" "traffic" "weather")
WINDOW_SIZES=(32 64 128)

for WINDOW_SIZE in "${WINDOW_SIZES[@]}"; do
    for CONFIG in "${CONFIGS[@]}"; do

        uv run main.py --name "train_${CONFIG}_${WINDOW_SIZE}" \
            --config_file "./config/${CONFIG}.yaml" --gpu 0 \
            --window_size "${WINDOW_SIZE}" --train

    done
done
