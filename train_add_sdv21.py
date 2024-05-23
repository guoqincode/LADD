import argparse
import logging
import os

from core.network.unet_D import Discriminator
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from packaging import version
from torch.utils.data import default_collate
from tqdm.auto import tqdm

import diffusers
from diffusers import (
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
import wandb
from core.data.dataset import ADDDataset

from core.utils import predicted_origin


logger = get_logger(__name__)



def log_validation(unet, accelerator, scheduler, args):
    logger.info('Running validation... ')

    unet = accelerator.unwrap_model(unet)

    pipe = DiffusionPipeline.from_pretrained(args.base_model,
                                    unet=unet,
                                    scheduler=scheduler,
                                    requires_safety_checker=False,
                                    safety_checker=None)
    pipe.to('cuda')

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    validation_prompts = [
        'A beautiful dessert waiting to be shared by two people',
        'A shot of an elderly man inside a kitchen',
        'Two tall giraffes graze on bushes in an open field',
        'A double decker bus driving down the road',
        'Two young men standing around a table in front of a chalk board'
    ]

    image_logs = []

    for _, prompt in enumerate(validation_prompts):
        images = []
        for i in range(len(validation_prompts)):

            with torch.autocast(accelerator.device.type):
                image = pipe(validation_prompts[i], num_inference_steps=1, timesteps=[999], generator=generator).images[0]

            images.append(image)


        image_logs.append({'validation_prompt': prompt, 'images': images})

    for tracker in accelerator.trackers:
        if tracker.name == 'wandb':
            formatted_images = []

            for log in image_logs:
                images = log['images']
                validation_prompt = log['validation_prompt']
                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({f'validation': formatted_images})
        else:
            logger.warn(f'image logging not implemented for {tracker.name}')
        torch.cuda.empty_cache()

        return image_logs




def parse_args():
    parser = argparse.ArgumentParser(description='Training LADD')
    # ----------MODEL INFO----------
    parser.add_argument(
        '--base_model',
        type=str,
        default='stabilityai/stable-diffusion-2-1-base',
        help='Model identifier from huggingface.co/models.',
    )

    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help=(
            'Path to load a previously trained checkpoint'
        ),
    )

    # ----------TRAINING OPTIONS----------
    parser.add_argument('--seed', type=int, default=None, help='A seed for reproducible training.')

    parser.add_argument(
        '--checkpointing_steps',
        type=int,
        default=500,
        help=(
            'Save checkpoints at every X updates'
        ),
    )

    parser.add_argument(
        '--validation_steps',
        type=int,
        default=200,
        help='Run validation every X steps.',
    )

    parser.add_argument(
        '--train_batch_size', type=int, default=16, help='Batch size (per device) for the training dataloader.'
    )

    parser.add_argument(
        '--max_train_steps',
        type=int,
        default=100000,
        help='Total number of training steps to perform.',
    )

    parser.add_argument(
        '--mixed_precision',
        type=str,
        default=None,
        choices=['no', 'fp16', 'bf16'],
        help=(
            'Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >='
            ' 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the'
            ' flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.'
        ),
    )
    
    parser.add_argument(
        '--enable_xformers_memory_efficient_attention', action='store_true', help='Whether or not to use xformers.'
    )

    parser.add_argument(
        '--gradient_checkpointing',
        action='store_true',
        help='Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.',
    )

    parser.add_argument(
        '--recon_lambda',
        type=float,
        default=1.0,
        help=('loss weight for recon')
    )

    parser.add_argument(
        '--multiscale_D',
        action='store_true',
        help='Whether to use multi-scale Discriminator. Extra heads from intermediate features',
    )

    parser.add_argument(
        '--misaligned_pairs_D',
        action='store_true',
        help='Whether to use mis aligned pairs for Discriminator. Pair some real images with misaligned prompts to enforce text-image alignment abilities of Discriminator.',
    )

    parser.add_argument(
        '--D_ts',
        type=str,
        default='10, 249, 499, 749',
        help='Timestep choices for Discriminator'
    )

    parser.add_argument(
        '--num_ts',
        type=int,
        default=1,
        help=('Number of time steps to sample from the original 1000 time steps for training G. We train one-step model by default')
    )

    parser.add_argument(
        '--zero_snr',
        action='store_true',
        help='Whether to enforce zero SNR, meaning set SNR to 0 for ts=999. Generating images from pure noises',
    )

    parser.add_argument('--local_rank', type=int, default=-1, help='For distributed training: local_rank')

    # ----------DATA INFO----------    
    parser.add_argument(
        '--dataset_root',
        type=str,
        default=None,
        help=(
            'root folder for dataset'
        ),
    )

    parser.add_argument(
        '--data_pkl_name',
        type=str,
        choices=['summary.pkl', 'summary_llava.pkl', 'summary_noise_img_pair.pkl'],
        help=(
            'pkl file for loading data paths'
        ),
    )

    parser.add_argument(
        '--img_size',
        type=int,
        default=512,
        help=(
            'input image size'
        ),
    )

    parser.add_argument(
        '--dataloader_num_workers',
        type=int,
        default=0,
        help=(
            'Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.'
        ),
    )

    # ----------TRACKER INFO----------
    parser.add_argument(
        '--report_to',
        type=str,
        default='tensorboard',
        help=(
            'The integration to report the results and logs to. Supported platforms are \'tensorboard\''
            ' (default), \'wandb\' and \'comet_ml\'. Use \'all\' to report to all integrations.'
        ),
    )

    parser.add_argument('--exp_name', type=str, default='default_exp_name', help='identify exp name')

    parser.add_argument(
        '--project_name',
        type=str,
        default='sdv2.1_add',
        help=(
            'The `project_name` argument passed to Accelerator.init_trackers for'
        ),
    )

    # ----------OPTIMIZER INFO----------   
    parser.add_argument(
        '--G_lr',
        type=float,
        default=1e-6,
        help='learning rate for generator',
    )
    parser.add_argument(
        '--D_lr',
        type=float,
        default=1e-6,
        help='learning rate for discriminator',
    )
    parser.add_argument(
        '--lr_scheduler',
        type=str,
        default='constant',
        help=(
            'The scheduler type to use. Choose between [\'linear\', \'cosine\', \'cosine_with_restarts\', \'polynomial\','
            ' \'constant\', \'constant_with_warmup\']'),
    )
    parser.add_argument(
        '--lr_warmup_steps', type=int, default=500, help='Number of steps for the warmup in the lr scheduler.'
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of updates steps to accumulate before performing a backward/update pass.',
    )

    parser.add_argument('--G_adam_beta1', type=float, default=0.9, help='The beta1 parameter for the Adam optimizer in G.')
    parser.add_argument('--G_adam_beta2', type=float, default=0.999, help='The beta2 parameter for the Adam optimizer in G.')
    parser.add_argument('--G_adam_wd', type=float, default=1e-2, help='Weight decay to use in G.')
    parser.add_argument('--G_adam_epsilon', type=float, default=1e-08, help='Epsilon value for the Adam optimizer')

    parser.add_argument('--D_adam_beta1', type=float, default=0.9, help='The beta1 parameter for the Adam optimizer in G.')
    parser.add_argument('--D_adam_beta2', type=float, default=0.999, help='The beta2 parameter for the Adam optimizer in G.')
    parser.add_argument('--D_adam_wd', type=float, default=1e-2, help='Weight decay to use in G.')
    parser.add_argument('--D_adam_epsilon', type=float, default=1e-08, help='Epsilon value for the Adam optimizer')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='Max gradient norm.')   



    args = parser.parse_args()
    env_local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main(args):

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        split_batches=True,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(project_name=args.project_name, config=args, init_kwargs={'wandb': {'name': args.exp_name}})
        wandb_tracker = accelerator.get_tracker('wandb', unwrap=True)
        ckpt_dir = os.path.join(wandb_tracker.dir, '..', 'ckpts') # parent folder of folder 'files'
        os.makedirs(ckpt_dir, exist_ok=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.base_model, subfolder='scheduler'
    )

    if args.zero_snr:
        def add_noise(
            self,
            original_samples: torch.FloatTensor,
            noise: torch.FloatTensor,
            timesteps: torch.IntTensor,
        ) -> torch.FloatTensor:

            # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
            # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
            # for the subsequent add_noise calls
            self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
            alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
            timesteps = timesteps.to(original_samples.device)

            sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
            sqrt_alpha_prod[timesteps==999] = 0
            sqrt_alpha_prod = sqrt_alpha_prod.flatten()
            while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
                sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

            sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
            sqrt_one_minus_alpha_prod[timesteps==999] = 1
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
            while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

            noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
            return noisy_samples

        noise_scheduler.add_noise = add_noise.__get__(noise_scheduler, DDPMScheduler)


    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)

    disc = Discriminator(args.base_model, args.multiscale_D)
    disc.train()

    unet = UNet2DConditionModel.from_pretrained(
        args.base_model, subfolder='unet')
    unet.train()

    if args.resume_from_checkpoint is not None:
        print('Loading models from: ', args.base_model)
        disc.load_state_dict(torch.load(os.path.join(args.base_model, 'disc.bin')), strict=True)
        load_model = UNet2DConditionModel.from_pretrained(os.path.join(args.base_model, 'unet'))
        unet.load_state_dict(load_model.state_dict())
        del load_model


    # Also move the alpha and sigma noise schedules to accelerator.device.
    alpha_schedule = alpha_schedule.to(accelerator.device)
    sigma_schedule = sigma_schedule.to(accelerator.device)


    # 12. Enable optimizations
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse('0.0.16'):
                logger.warn(
                    'xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details.'
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError('xformers is not available. Make sure it is installed correctly')


    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        disc.unet.enable_gradient_checkpointing()


    optimizer_G = torch.optim.AdamW(
        unet.parameters(),
        lr=args.G_lr,
        betas=(args.G_adam_beta1, args.G_adam_beta2),
        weight_decay=args.G_adam_wd,
        eps=args.G_adam_epsilon,
    )

    optimizer_D = torch.optim.AdamW(
        disc.parameters(),
        lr=args.D_lr,
        betas=(args.D_adam_beta1, args.D_adam_beta2),
        weight_decay=args.D_adam_wd,
        eps=args.D_adam_epsilon,
    )



    train_dataset = ADDDataset(args.dataset_root,
                            args.data_pkl_name)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=default_collate,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_G,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, disc, optimizer_G, optimizer_D, lr_scheduler = accelerator.prepare(
        unet, disc, optimizer_G, optimizer_D, lr_scheduler
    )


    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info('***** Running training *****')
    logger.info(f'  Num batches each epoch = {len(train_dataset) // args.train_batch_size}')
    logger.info(f'  Instantaneous batch size per device = {args.train_batch_size}')
    logger.info(f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}')
    logger.info(f'  Gradient Accumulation steps = {args.gradient_accumulation_steps}')
    logger.info(f'  Total optimization steps = {args.max_train_steps}')
    global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc='Steps',
        disable=not accelerator.is_local_main_process,
    )

    # Set initial phase to G. We switch between G and D
    phase = 'G'

    ts_D_choices = torch.tensor([int(i) for i in args.D_ts.split(',')], device=accelerator.device).long()

    timestep_list = np.linspace(1000, 0, num=args.num_ts, endpoint=False) - 1
    timestep_list = torch.tensor(timestep_list).long()
    sqrt_alpha_prod_list = noise_scheduler.alphas_cumprod[timestep_list] ** 0.5
    sqrt_one_minus_alpha_prod_list = (1 - noise_scheduler.alphas_cumprod[timestep_list]) ** 0.5
    timestep_list = timestep_list.to(accelerator.device)
    sqrt_alpha_prod_list = sqrt_alpha_prod_list.to(accelerator.device)
    sqrt_one_minus_alpha_prod_list = sqrt_one_minus_alpha_prod_list.to(accelerator.device)

    logger.info(f'timesteps for D: {ts_D_choices}')
    logger.info(f'timesteps for G: {timestep_list}')

    while True: #terminate training according to iters
        for step, batch in enumerate(train_dataloader):
            latents, noises, text_embs = batch
            latents = latents.to(accelerator.device, non_blocking=True)
            noises = noises.to(accelerator.device, non_blocking=True)
            for k in text_embs:
                text_embs[k] = text_embs[k].to(accelerator.device, non_blocking=True)

            bsz = latents.shape[0]

            ts_indices = torch.randint(0, args.num_ts, (bsz, ))
            timesteps = timestep_list[ts_indices].long()
            noisy_model_input = noise_scheduler.add_noise(latents, noises, timesteps)
            
            if phase == 'G':
                with accelerator.accumulate(unet):

                    disc.eval()
                    unet.train()
                
                    noise_pred = unet(
                        noisy_model_input,
                        timesteps,
                        **text_embs,
                    ).sample

                    pred_x_0 = predicted_origin(
                        noise_pred,
                        timesteps,
                        noisy_model_input,
                        noise_scheduler.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )
                    
                    # add noise to generated latents and feed them to D
                    timesteps_D = ts_D_choices[torch.randint(0, len(ts_D_choices), (bsz, ), device=accelerator.device)]
                    noised_predicted_x0 = noise_scheduler.add_noise(pred_x_0, torch.randn_like(latents), timesteps_D)
                    
                    # adv loss
                    pred_fake = disc(noised_predicted_x0, timesteps_D, **text_embs)
                    adv_loss = F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))

                    #recon loss
                    recon_loss = F.smooth_l1_loss(pred_x_0, latents)

                    #total loss
                    loss = adv_loss + recon_loss * args.recon_lambda

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        #switch phase to D
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                        phase = 'D'
                    optimizer_G.step()
                    lr_scheduler.step()
                    optimizer_G.zero_grad(set_to_none=True)
                    optimizer_D.zero_grad(set_to_none=True)
                
                logs = {'adv_loss': adv_loss.detach().item(), 
                        'recon_loss': recon_loss.detach().item(),
                        'lr': lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
            
            elif phase == 'D':
                with accelerator.accumulate(disc):
                    disc.train()
                    unet.eval()

                    with torch.no_grad():
                        noise_pred = unet(
                            noisy_model_input,
                            timesteps,
                            **text_embs,
                        ).sample

                        pred_x_0 = predicted_origin(
                            noise_pred,
                            timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )

                    timesteps_D_fake = ts_D_choices[torch.randint(0, len(ts_D_choices), (bsz, ), device=accelerator.device)]
                    timesteps_D_real = ts_D_choices[torch.randint(0, len(ts_D_choices), (bsz, ), device=accelerator.device)]
                    noised_predicted_x0 = noise_scheduler.add_noise(pred_x_0, torch.randn_like(latents), timesteps_D_fake)
                    noised_latents = noise_scheduler.add_noise(latents, torch.randn_like(latents), timesteps_D_real)

                    if args.misaligned_pairs_D and bsz > 1:
                        shifted_latents = torch.roll(latents, 1, 0)
                        timesteps_D_shifted_pairs = ts_D_choices[torch.randint(0, len(ts_D_choices), (bsz, ), device=accelerator.device)]
                        noised_shifted_latents = noise_scheduler.add_noise(shifted_latents, torch.randn_like(shifted_latents), timesteps_D_shifted_pairs)

                        noised_predicted_x0 = torch.concat([noised_predicted_x0, noised_shifted_latents], dim=0)
                        timesteps_D_fake = torch.concat([timesteps_D_fake, timesteps_D_shifted_pairs])
                        # prompt_embeds_fake = torch.concat([text_embs, text_embs], dim=0)
                        prompt_embeds_fake = {}
                        for k in text_embs:
                            prompt_embeds_fake[k] = torch.concat([text_embs[k], text_embs[k]], dim=0)
                    else:
                        prompt_embeds_fake = text_embs
                        

                    
                    pred_fake = disc(noised_predicted_x0, timesteps_D_fake, **prompt_embeds_fake)
                    pred_true = disc(noised_latents, timesteps_D_real, **text_embs)
                    
                    #calculate losses for fake and real data
                    loss_gen = F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
                    loss_real = F.binary_cross_entropy_with_logits(pred_true, torch.ones_like(pred_true))
                    D_loss = loss_gen + loss_real

                    accelerator.backward(D_loss)

                    if accelerator.sync_gradients:
                        #swith back to phase G and add global step by one.
                        accelerator.clip_grad_norm_(disc.parameters(), args.max_grad_norm)
                        phase = 'G'
                        global_step += 1
                        progress_bar.update(1)
                    
                    optimizer_D.step()
                    optimizer_G.zero_grad(set_to_none=True)
                    optimizer_D.zero_grad(set_to_none=True)
                
                logs = {'D_loss': D_loss.detach().item(), 'loss_gen': loss_gen.detach().item(), 
                        'loss_real': loss_real.detach().item(), 'lr': lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                        
            if accelerator.sync_gradients:
                if accelerator.is_main_process:
                    # check if we need to save the ckpts
                    if global_step and global_step % args.checkpointing_steps == 0 and phase == 'D':
                        save_path = os.path.join(ckpt_dir, f'checkpoint-{global_step}')
                        try:
                            accelerator.unwrap_model(unet).save_pretrained(os.path.join(save_path, 'unet'))
                            accelerator.unwrap_model(disc).save_pretrained(os.path.join(save_path, 'disc.bin'))
                        except Exception as e:
                            logger.info('error saving ckpts')
                            print(e)
                        logger.info(f'Saved state to {save_path}')

                    if global_step and global_step % args.validation_steps == 0 and phase == 'D':
                        log_validation(unet, accelerator, noise_scheduler, args)

            if global_step >= args.max_train_steps:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    accelerator.unwrap_model(unet).save_pretrained(os.path.join(save_path, 'unet'))
                    accelerator.unwrap_model(disc).save_pretrained(os.path.join(save_path, 'disc.bin'))
                accelerator.end_training()
                return

    


if __name__ == '__main__':
    args = parse_args()
    main(args)
