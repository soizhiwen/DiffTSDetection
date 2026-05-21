#!/bin/bash

CONFIGS=("electricity" "energy" "etth" "ettm" "exchange_rate" "fmri" "illness" "stock" "traffic" "weather")
WINDOW_SIZES=(64 128)
GENERATORS=("sssd" "tsdiff" "wavestitch")

for WINDOW_SIZE in "${WINDOW_SIZES[@]}"; do
    for CONFIG in "${CONFIGS[@]}"; do

        # uv run main.py --name "one_step_${CONFIG}_real_diffusionts_${WINDOW_SIZE}_1" \
        #     --config_file "./config/${CONFIG}.yaml" --gpu 0 \
        #     --window_size "${WINDOW_SIZE}" --mode 1 --one_step

        for GENERATOR in "${GENERATORS[@]}"; do

            uv run main.py --name "one_step_${CONFIG}_${GENERATOR}_diffusionts_${WINDOW_SIZE}_1" \
                --config_file "./config/${CONFIG}.yaml" --gpu 0 \
                --window_size "${WINDOW_SIZE}" --mode 1 --one_step \
                --data_root "../wavestitch-synth/datasets/${GENERATOR}/${CONFIG}_${WINDOW_SIZE}.npz"

        done
    done
done
