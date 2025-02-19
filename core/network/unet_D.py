from diffusers import UNet2DConditionModel
import torch.nn as nn
import torch
from typing import Union






def unet_forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        # added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        is_multiscale = False,
    ):
    

    # 0. center input if necessary
    if self.config.center_input_sample:
        sample = 2 * sample - 1.0

    # 1. time
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
        # This would be a good case for the `match` statement (Python 3.10+)
        is_mps = sample.device.type == "mps"
        if isinstance(timestep, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)

    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timesteps = timesteps.expand(sample.shape[0])

    t_emb = self.time_proj(timesteps)

    # `Timesteps` does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=sample.dtype)

    emb = self.time_embedding(t_emb, None)
    aug_emb = None


    if self.config.addition_embed_type == "text":
        aug_emb = self.add_embedding(encoder_hidden_states)
    


    emb = emb + aug_emb if aug_emb is not None else emb

    if self.time_embed_act is not None:
        emb = self.time_embed_act(emb)

    if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
        encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)


    # 2. pre-process
    sample = self.conv_in(sample)



    # 3. down
    lora_scale = 1.0

    feat_list = []

    down_block_res_samples = (sample,)
    for blk_ind, downsample_block in enumerate(self.down_blocks):
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            # For t2i-adapter CrossAttnDownBlock2D
            additional_residuals = {}
        
            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=None,
                cross_attention_kwargs=None,
                encoder_attention_mask=None,
                **additional_residuals,
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb, scale=lora_scale)
        down_block_res_samples += res_samples
        if is_multiscale and blk_ind <= 1:
            feat_list.append(sample)

    # 4. mid
    if self.mid_block is not None:
        if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=None,
                cross_attention_kwargs=None,
                encoder_attention_mask=None,
            )
        else:
            sample = self.mid_block(sample, emb)

    feat_list.append(sample)
    return feat_list


class Discriminator(nn.Module):
    def __init__(self, pretrained_path, is_multiscale=False):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_path, subfolder="unet")
        self.unet.forward = unet_forward
        self.unet.up_blocks = None
        self.unet.conv_out = None
        self.unet.conv_norm_out = None
        self.is_multiscale = is_multiscale

        self.heads = []
        if is_multiscale:
            channel_list = [320, 640, 1280]
        else:
            channel_list = [1280]

        for feat_c in channel_list:
            self.heads.append(nn.Sequential(nn.GroupNorm(32, feat_c, eps=1e-05, affine=True),
                                        nn.Conv2d(feat_c, feat_c//4, 4, 2, 2),
                                        nn.SiLU(),
                                        nn.Conv2d(feat_c//4,1,1,1,0)
                                            ))
        self.heads = nn.ModuleList(self.heads)

        
    def forward(self, latent, timesteps, encoder_hidden_states):
        feat_list = self.unet.forward(self.unet, latent, timesteps, encoder_hidden_states, is_multiscale=self.is_multiscale)

        res_list = []
        for cur_feat, cur_head in zip(feat_list, self.heads):
            cur_out = cur_head(cur_feat)
            res_list.append(cur_out.reshape(cur_out.shape[0], -1))
        
        concat_res = torch.cat(res_list, dim=1)
        
        return concat_res



    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)