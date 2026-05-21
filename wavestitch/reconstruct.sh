#!/bin/bash

CONFIGS=("electricity" "energy" "etth" "ettm" "exchange_rate" "fmri" "illness" "stock" "traffic" "weather")
WINDOW_SIZES=(32 64 128)
VARIANTS=("real" "sssd" "tsdiff" "wavestitch" "diffusionts")

for WINDOW_SIZE in "${WINDOW_SIZES[@]}"; do
    for CONFIG in "${CONFIGS[@]}"; do

        uv run recon_sssd.py -d "${CONFIG}" -window_size "${WINDOW_SIZE}" -variant "real" -mode 0
        uv run recon_sssd.py -d "${CONFIG}" -window_size "${WINDOW_SIZE}" -variant "sssd" -mode 0

        for VARIANT in "${VARIANTS[@]}"; do

            uv run recon_sssd.py -d "${CONFIG}" -window_size "${WINDOW_SIZE}" -variant "${VARIANT}" -mode 1

        done
    done
done
