import pickle
import os
import random
import datetime
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='args for creating summary')

parser.add_argument(
        '--root_folder',
        type=str,
        default='../generated_data',
        help=(
            'root folder for generated data'
        ),
    )

parser.add_argument(
        '--summary_file_name',
        type=str,
        default='summary.pkl',
        help=(
            'summary file name'
        ),
    )

args = parser.parse_args()


latent_folder = 'latents'
txt_emb_folder = 'txt_embs'
noise_folder = 'noises'

res_list = []
for fname in tqdm(os.listdir(os.path.join(args.root_folder, latent_folder))):
    latent_rel_path = os.path.join(latent_folder, fname)
    txt_emb_rel_path = os.path.join(txt_emb_folder, fname)
    noise_rel_path = os.path.join(noise_folder, fname)

    if os.path.exists(os.path.join(args.root_folder, latent_rel_path)) and \
        os.path.exists(os.path.join(args.root_folder, txt_emb_rel_path)) and \
        os.path.exists(os.path.join(args.root_folder, noise_rel_path)):
        res_list.append((latent_rel_path, 
                        noise_rel_path, txt_emb_rel_path))
    else:
        print('skipped for missing data')

with open(os.path.join(args.root_folder, args.summary_file_name), 'wb') as f:
    pickle.dump(res_list, f)

print(f'done. found {len(res_list)} pairs')
