import os
import random
import datetime
import torch
from diffusers import DiffusionPipeline, DDIMScheduler, PixArtSigmaPipeline, Transformer2DModel
import argparse

parser = argparse.ArgumentParser(description='args for generating synthetic data')

parser.add_argument(
        '--prompt_path',
        type=str,
        default=None,
        required=True,
        help=(
            'a list containing prompts for generating images'
        ),
    )

parser.add_argument(
        '--base_model',
        type=str,
        default='stabilityai/stable-diffusion-2-1-base',
        help=(
            'base model identifier for huggingface'
        ),
    )

parser.add_argument(
        '--root_folder',
        type=str,
        default='../generated_data',
        help=(
            'root folder for generated data'
        ),
    )

parser.add_argument(
        '--cfg',
        type=float,
        default=3.0,
        help=(
            'guidance scale. Default: 3.0'
        ),
    )

parser.add_argument(
        '--noise_c',
        type=int,
        default=4,
        help=(
            'channel width for noise'
        ),
    )

parser.add_argument(
        '--noise_size',
        type=int,
        default=64,
        help=(
            'spatial size for noise. width and height are same.'
        ),
    )

parser.add_argument(
        '--save_precision',
        type=str,
        default='fp16',
        help=(
            'data saving precision. choose between [fp16, fp32]'
        ),
        choices=['fp32', 'fp16'],
    )

args = parser.parse_args()

if args.save_precision == 'fp32':
    weight_type = torch.float32
elif args.save_precision == 'fp16':
    weight_type = torch.float16

os.makedirs(args.root_folder, exist_ok=True)

latent_folder = os.path.join(args.root_folder, 'latents')
txt_emb_folder = os.path.join(args.root_folder, 'txt_embs')
noise_folder = os.path.join(args.root_folder, 'noises')

os.makedirs(latent_folder, exist_ok=True)
os.makedirs(txt_emb_folder, exist_ok=True)
os.makedirs(noise_folder, exist_ok=True)

with open(args.prompt_path, 'r') as f:
    lines = f.readlines()

anno_list = []
for ind, cur_l in enumerate(lines):
    anno_list.append((str(ind), cur_l))


random.seed(datetime.datetime.now().timestamp() * 10000)
random.shuffle(anno_list)



if args.base_model == 'stabilityai/stable-diffusion-2-1-base':
    pipe = DiffusionPipeline.from_pretrained(args.base_model)
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config)

    def encoding_func(prompt):
        inputs = pipe.tokenizer(
            prompt, max_length=pipe.tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors="pt"
                ).input_ids
        encoder_hidden_states = pipe.text_encoder(inputs.cuda(), return_dict=False)[0].squeeze().to(weight_type).cpu()
        return {'encoder_hidden_states': encoder_hidden_states}
elif args.base_model in ['PixArt-alpha/PixArt-Sigma-XL-2-1024-MS',
                        'PixArt-alpha/PixArt-Sigma-XL-2-512-MS']:
    transformer = Transformer2DModel.from_pretrained(
    args.base_model, 
    subfolder='transformer', 
    use_safetensors=True,
)
    pipe = PixArtSigmaPipeline.from_pretrained(
    'PixArt-alpha/PixArt-Sigma-XL-2-1024-MS',
    transformer=transformer,
    use_safetensors=True,
    )

    def encoding_func(prompt):
        prompt = pipe._text_preprocessing(prompt, clean_caption=True)
        text_inputs = pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=300,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.to('cuda')
        prompt_embeds = pipe.text_encoder(text_input_ids.to('cuda'), attention_mask=prompt_attention_mask)[0]
        prompt_embeds = prompt_embeds.squeeze().to(weight_type).cpu()
        atten_mask = prompt_attention_mask.squeeze().cpu()
        return {'encoder_hidden_states': prompt_embeds, 
                'encoder_attention_mask': atten_mask}
else:
    raise Exception(f'model \'{args.base_model}\' not found')

pipe.to('cuda')

num_samples = len(anno_list)
for i in range(len(anno_list)):
    prompt = anno_list[i][1]
    basename = anno_list[i][0]+'.png'

    lock_path = os.path.join(args.root_folder, anno_list[i][0]+'.lock')
    latent_out_path = os.path.join(latent_folder, anno_list[i][0]+'.pth')
    txt_emb_out_path = os.path.join(txt_emb_folder, anno_list[i][0]+'.pth')
    noise_out_path = os.path.join(noise_folder, anno_list[i][0]+'.pth')
    if os.path.exists(latent_out_path) or os.path.exists(lock_path) :
        print('Skip sample %s' % basename)
        continue
    os.makedirs(lock_path, exist_ok=True)
    print('Generating sample %d/%d' % (i, num_samples))
    noise = torch.randn((1, args.noise_c, args.noise_size, args.noise_size))
    latent = pipe(prompt=prompt, guidance_scale=args.cfg, latents=noise.cuda(), output_type='latent').images
    emb_info = encoding_func(prompt)
    torch.save(latent.squeeze().to(weight_type).cpu(), latent_out_path)
    torch.save(noise.squeeze().to(weight_type).cpu(), noise_out_path)
    torch.save(emb_info, txt_emb_out_path)
    os.rmdir(lock_path)

    