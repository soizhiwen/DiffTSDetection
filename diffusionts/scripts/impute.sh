#!/bin/bash

CONFIGS=("electricity" "energy" "etth" "ettm" "exchange_rate" "fmri" "illness" "stock" "traffic" "weather")
WINDOW_SIZES=(32)
GENERATORS=("sssd" "tsdiff" "wavestitch")

for WINDOW_SIZE in "${WINDOW_SIZES[@]}"; do
    for CONFIG in "${CONFIGS[@]}"; do

        # uv run main.py --name "impute_${CONFIG}_real_diffusionts_${WINDOW_SIZE}_0" \
        #     --config_file "./config/${CONFIG}.yaml" --gpu 0 \
        #     --window_size "${WINDOW_SIZE}" --missing_ratio 0.5 --mode 0 --impute

        for GENERATOR in "${GENERATORS[@]}"; do

            uv run main.py --name "impute_${CONFIG}_${GENERATOR}_diffusionts_${WINDOW_SIZE}_0" \
                --config_file "./config/${CONFIG}.yaml" --gpu 0 \
                --window_size "${WINDOW_SIZE}" --missing_ratio 0.5 --mode 0 --impute \
                --data_root "../wavestitch-synth/datasets/${GENERATOR}/${CONFIG}_${WINDOW_SIZE}.npz"

        done
    done
done
