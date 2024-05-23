PROMPT_PATH='../generate_images/random_data.txt'
OUT_FOLDER='../generated_data_pixart'
MODEL_ID='PixArt-alpha/PixArt-Sigma-XL-2-512-MS'

available_gpus=(0 1 2 3 4 5 6 7)
for gpu in "${available_gpus[@]}"; do
CUDA_VISIBLE_DEVICES=${gpu} python core/tools/gen_synthetic_data.py \
                            --prompt_path $PROMPT_PATH \
                            --root_folder $OUT_FOLDER \
                            --base_model $MODEL_ID &
done

# after generating data, run this to create a summary file
# python core/tools/create_summary.py --root_folder $OUT_FOLDER
