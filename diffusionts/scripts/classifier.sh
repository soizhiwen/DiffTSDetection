#!/bin/bash

CONFIGS=("electricity" "energy" "etth" "ettm" "exchange_rate" "fmri" "illness" "stock" "traffic" "weather")
WINDOW_SIZES=(32 64 128)
GENERATORS=("sssd" "tsdiff" "diffusionts" "wavestitch")

# for WINDOW_SIZE in "${WINDOW_SIZES[@]}"; do
#     for CONFIG in "${CONFIGS[@]}"; do
#         uv run disjointcnn.py --dataset "${CONFIG}" --window_size "${WINDOW_SIZE}" --train
#         for GENERATOR in "${GENERATORS[@]}"; do
#             uv run disjointcnn.py --dataset "${CONFIG}" --window_size "${WINDOW_SIZE}" --generator "${GENERATOR}" --eval
#         done
#     done
# done

for WINDOW_SIZE in "${WINDOW_SIZES[@]}"; do
    for CONFIG in "${CONFIGS[@]}"; do
        uv run classifier.py --dataset "${CONFIG}" --window_size "${WINDOW_SIZE}" --train
        for GENERATOR in "${GENERATORS[@]}"; do
            uv run classifier.py --dataset "${CONFIG}" --window_size "${WINDOW_SIZE}" --generator "${GENERATOR}" --eval
        done
    done
done
