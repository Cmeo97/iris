from einops import rearrange
import numpy as np
from PIL import Image
import torch
from torchvision.utils import save_image


@torch.no_grad()
def make_reconstructions_from_batch(batch, save_dir, epoch, tokenizer):
    check_batch(batch)

    original_frames = tensor_to_np_frames(rearrange(batch['observations'], 'b t c h w -> b t h w c'))
    all = [original_frames]

    rec_frames = generate_reconstructions_with_tokenizer(batch, tokenizer)
    all.append(rec_frames)

    for i, image in enumerate(map(Image.fromarray, np.concatenate(list(np.concatenate((original_frames, rec_frames), axis=-2)), axis=-3))):
        image.save(save_dir / f'epoch_{epoch:03d}_t_{i:03d}.png')

    return

@torch.no_grad()
def make_reconstructions_with_slots_from_batch(batch, save_dir, epoch, tokenizer):
    # check_batch(batch)

    inputs = rearrange(batch['observations'], 'b t c h w -> (b t) c h w')
    outputs = reconstruct_through_tokenizer_with_slots(inputs, tokenizer)
    b, t, _, _, _ = batch['observations'].size()
    recons, colors, masks = outputs
    recons = rearrange(recons, '(b t) c h w -> b t c h w', b=b, t=t)
    colors = rearrange(colors, '(b t) k c h w -> b t k c h w', b=b, t=t)
    masks = rearrange(masks, '(b t) k c h w -> b t k c h w', b=b, t=t)

    save_image_with_slots(batch['observations'], recons, colors, masks, save_dir, epoch)

def save_image_with_slots(observations, recons, colors, masks, save_dir, epoch, suffix='sample'):
    b, t, _, _, _ = observations.size()

    for i in range(b):
        obs = observations[i].cpu() # (t c h w)
        recon = recons[i].cpu() # (t c h w)

        full_plot = torch.cat([obs.unsqueeze(1), recon.unsqueeze(1)], dim=1) # (t 2 c h w)
        color = colors[i].cpu()
        mask = masks[i].cpu()
        subimage = color * mask
        mask = mask.repeat(1,1,3,1,1)
        full_plot = torch.cat([full_plot, mask, subimage], dim=1) #(T,2+K+K,3,D,D)
        full_plot = full_plot.permute(1, 0, 2, 3, 4).contiguous()  # (H,W,3,D,D)
        full_plot = full_plot.view(-1, 3, 64, 64)  # (H*W, 3, D, D)

        save_image(full_plot, save_dir / f'epoch_{epoch:03d}_{suffix}_{i:03d}.png', nrow=t)

    return


def check_batch(batch):
    assert sorted(batch.keys()) == ['actions', 'ends', 'mask_padding', 'observations', 'rewards']
    b, t, _, _, _ = batch['observations'].shape  # (B, T, C, H, W)
    assert batch['actions'].shape == batch['rewards'].shape == batch['ends'].shape == batch['mask_padding'].shape == (b, t)


def tensor_to_np_frames(inputs):
    check_float_btw_0_1(inputs)
    return inputs.mul(255).cpu().numpy().astype(np.uint8)


def check_float_btw_0_1(inputs):
    assert inputs.is_floating_point() and (inputs >= 0).all() and (inputs <= 1).all()


@torch.no_grad()
def generate_reconstructions_with_tokenizer(batch, tokenizer):
    check_batch(batch)
    inputs = rearrange(batch['observations'], 'b t c h w -> (b t) c h w')
    outputs = reconstruct_through_tokenizer(inputs, tokenizer)
    b, t, _, _, _ = batch['observations'].size()
    outputs = rearrange(outputs, '(b t) c h w -> b t h w c', b=b, t=t)
    rec_frames = tensor_to_np_frames(outputs)
    return rec_frames


@torch.no_grad()
def reconstruct_through_tokenizer(inputs, tokenizer):
    check_float_btw_0_1(inputs)
    reconstructions = tokenizer.encode_decode(inputs, should_preprocess=True, should_postprocess=True)
    return torch.clamp(reconstructions, 0, 1)

@torch.no_grad()
def generate_reconstructions_and_slots_with_tokenizer(batch, tokenizer):
    check_batch(batch)
    inputs = rearrange(batch['observations'], 'b t c h w -> (b t) c h w')
    outputs = reconstruct_through_tokenizer_with_slots(inputs, tokenizer)
    b, t, _, _, _ = batch['observations'].size()
    outputs = rearrange(outputs, '(b t) c h w -> b t h w c', b=b, t=t)
    rec_frames = tensor_to_np_frames(outputs)
    return rec_frames


@torch.no_grad()
def reconstruct_through_tokenizer_with_slots(inputs, tokenizer):
    check_float_btw_0_1(inputs)
    reconstructions, colors, masks = tokenizer.encode_decode_slots(inputs, should_preprocess=True, should_postprocess=True)
    return torch.clamp(reconstructions, 0, 1), colors, masks
