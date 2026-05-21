#!/bin/bash

CONFIGS=("electricity" "energy" "etth" "ettm" "exchange_rate" "fmri" "illness" "stock" "traffic" "weather")
WINDOW_SIZES=(32 64 128)

for WINDOW_SIZE in "${WINDOW_SIZES[@]}"; do
    for CONFIG in "${CONFIGS[@]}"; do

        uv run training_SSSD.py -d "${CONFIG}" -window_size "${WINDOW_SIZE}" -epochs 300
        uv run training_wavestitch.py -d "${CONFIG}" -window_size "${WINDOW_SIZE}" -epochs 300
        uv run training_tsdiff.py -d "${CONFIG}" -window_size "${WINDOW_SIZE}" -epochs 300

    done
done
