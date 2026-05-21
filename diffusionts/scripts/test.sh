#!/bin/bash

CONFIG="stock"
WINDOW_SIZE=32
GENERATOR="diffusionts"

uv run main.py --name "train_${CONFIG}_${WINDOW_SIZE}" \
    --config_file "./config/${CONFIG}.yaml" --gpu 0 \
    --window_size "${WINDOW_SIZE}" --train --output "outputs_test"

uv run main.py --name "sample_${CONFIG}_${WINDOW_SIZE}" \
    --config_file "./config/${CONFIG}.yaml" --gpu 0 \
    --window_size "${WINDOW_SIZE}" --sample --output "outputs_test"

uv run main.py --name "recon_${CONFIG}_real_${WINDOW_SIZE}" \
    --config_file "./config/${CONFIG}.yaml" --gpu 0 \
    --window_size "${WINDOW_SIZE}" --reconstruct --output "outputs_test"

uv run main.py --name "recon_${CONFIG}_${GENERATOR}_${WINDOW_SIZE}" \
    --config_file "./config/${CONFIG}.yaml" --gpu 0 \
    --window_size "${WINDOW_SIZE}" --reconstruct --output "outputs_test" \
    --data_root "./datasets/${GENERATOR}/${CONFIG}_${WINDOW_SIZE}.npz"

uv run main.py --name "impute_${CONFIG}_real_${WINDOW_SIZE}" \
    --config_file "./config/${CONFIG}.yaml" --gpu 0 \
    --window_size "${WINDOW_SIZE}" --missing_ratio 0.5 --impute --output "outputs_test"

uv run main.py --name "impute_${CONFIG}_${GENERATOR}_${WINDOW_SIZE}" \
    --config_file "./config/${CONFIG}.yaml" --gpu 0 \
    --window_size "${WINDOW_SIZE}" --missing_ratio 0.5 --impute --output "outputs_test" \
    --data_root "./datasets/${GENERATOR}/${CONFIG}_${WINDOW_SIZE}.npz"
