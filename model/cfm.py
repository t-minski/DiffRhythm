# Copyright (c) 2025 ASLP-LAB
#               2025 Ziqian Ning   (ningziqian@mail.nwpu.edu.cn)
#               2025 Huakang Chen  (huakang@mail.nwpu.edu.cn)
#               2025 Guobin Ma     (guobin.ma@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" This implementation is adapted from github repo:
    https://github.com/SWivid/F5-TTS.
"""
from __future__ import annotations

import jaxtyping as jx
from typing import Literal
from beartype import beartype
import torch as th

from typing import Callable
from random import random

import torch
from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from torchdiffeq import odeint

from model.utils import (
    exists,
    list_str_to_idx,
    list_str_to_tensor,
    lens_to_mask,
    mask_from_frac_lengths,
)

def custom_mask_from_start_end_indices(
    seq_len: int["b"],  # noqa: F821
    latent_pred_segments,
    device,
    max_seq_len
):
    max_seq_len = max_seq_len
    seq = torch.arange(max_seq_len, device=device).long()

    res_mask = torch.zeros(max_seq_len, device=device, dtype=torch.bool)
    
    for start, end in latent_pred_segments:
        start = start.unsqueeze(0)
        end = end.unsqueeze(0)
        start_mask = seq[None, :] >= start[:, None]
        end_mask = seq[None, :] < end[:, None]
        res_mask = res_mask | (start_mask & end_mask)
    
    return res_mask

class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            method="euler"
        ),
        odeint_options: dict = dict(
            min_step=0.05
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        style_drop_prob=0.1,
        lrc_drop_prob=0.1,
        num_channels=None,
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
        max_frames=2048
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.style_drop_prob = style_drop_prob
        self.lrc_drop_prob = lrc_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs
        
        self.odeint_options = odeint_options

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map
        
        self.max_frames = max_frames

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    @jx.jaxtyped(typechecker=beartype)
    def sample(
        self,
        cond: jx.Float[th.Tensor, "B N D"] | jx.Float[th.Tensor, "B N"],
        text: jx.Int[th.Tensor, "B N"] | list[str],
        duration: int | jx.Int[th.Tensor, "B"],
        *,
        style_prompt: jx.Float[th.Tensor, "B S"] | None = None,
        negative_style_prompt: jx.Float[th.Tensor, "B S"] | None = None,
        lens: jx.Int[th.Tensor, "B"] | None = None,
        steps: int = 32,
        cfg_strength: float = 4.0,
        sway_sampling_coef: float | None = None,
        seed: int | None = None,
        max_duration: int = 6144,
        vocoder: Callable[
            [jx.Float[th.Tensor, "B D N"]],
            jx.Float[th.Tensor, "B N"]
        ] | None = None,
        no_ref_audio: bool = False,
        duplicate_test: bool = False,
        t_inter: float = 0.1,
        edit_mask: jx.Bool[th.Tensor, "B N"] | None = None,
        start_time: jx.Float[th.Tensor, "B"] | None = None,
        latent_pred_segments: list[list[int]] | jx.Int[th.Tensor, "B 2"] | None = None,
        song_duration: jx.Float[th.Tensor, "B"] | None = None,
        batch_infer_num: int = 1,
        x0: jx.Float[th.Tensor, "B N D"] | None = None,
        mode: Literal[
            "vanilla-diffrhythm",
            "instrumental-to-vocals",
            "instrumental-condition"
        ] = "vanilla-diffrhythm"
    ):
        """
    Perform latent-space sampling with optional audio/text conditioning, style prompts,
    classifier-free guidance, and span-based inpainting/editing. Behavior depends on
    `mode` and `latent_pred_segments`.

    Args:
        cond (Tensor):
            Conditioning audio. Shape `[B, T, D]` for latent features or `[B, N]` for raw
            waveforms (which will be converted to mel-latents). Used as:
            • audio context outside editable spans
            • direct conditioning (if mode enables it, e.g. "instrumental-condition")
            • overwritten back into output outside editable spans (unless spans cover all T)

        text (Tensor | list[str]):
            Text conditioning. Either:
            • integer token tensor of shape `[B, Nt]`, or
            • list of strings (converted internally to token ids).

        duration (int | Tensor):
            Target sequence length in frames. If int, a scalar duration applied to all
            batch items. If per-item tensor `[B]`, defines variable-length generation.
            Clamped to `max_duration`. Also sets the maximum diffusion integration length.

        style_prompt (Tensor):
            Style-conditioning embedding of shape `[B, Ds]`. Repeated for
            `batch_infer_num` samples.

        negative_style_prompt (Tensor):
            Negative-prompt embedding used for classifier-free guidance. Same shape as
            `style_prompt`.

        lens (Tensor | None):
            Lengths of the conditioning audio (before padding). Shape `[B]`. Used to
            build masks for variable-length conditioning. If None, inferred from `cond`.

        steps (int):
            Number of ODE integration steps for the diffusion solver.

        cfg_strength (float):
            Classifier-free guidance scale. `0` disables CFG and uses the unconditional
            prediction only.

        sway_sampling_coef (float | None):
            Optional temporal warping of ODE timesteps for "sway" sampling, modifying
            the diffusion schedule.

        seed (int | None):
            Random seed for deterministic noise initialization. Applied *per duration*
            element.

        max_duration (int):
            Upper bound on allowed sequence length. Prevents allocating absurdly large
            latent tensors.

        vocoder (Callable | None):
            Optional vocoder that maps `[B, D, T]` latents → `[B, N]` waveform.
            If None, returns latents directly.

        no_ref_audio (bool):
            If True, zeroes out the audio condition entirely.

        duplicate_test (bool):
            Special debugging mode that pads and blends `cond` into the initial noise
            state to inspect intermediate integration behavior.

        t_inter (float):
            Blend factor used when `duplicate_test=True`.

        edit_mask (Tensor | None):
            Boolean mask `[B, T]` restricting editable regions. Combined with span masks.

        start_time (Tensor):
            Per-sample starting time annotations for the model.

        latent_pred_segments (list[list[int]] | Tensor):
            List of `[start, end]` pairs defining the spans where new content is
            generated. These produce `fixed_span_mask`, controlling:
            • zeroing of audio conditioning inside spans
            • mixing of original vs. generated latents at the end.

        song_duration (Tensor):
            Per-sample total song duration used for positional conditioning.

        batch_infer_num (int):
            Number of parallel samples to draw from the same conditioning
            configuration (i.e., "N candidates from 1 prompt"). Repeats all inputs.

        x0 (Tensor | None):
            Optional initial latent state `[B, T, D]`. If provided, diffusion starts
            from `x0` instead of Gaussian noise. Used for instrumental→vocal bridging.
            Requires mode to be "instrumental-to-vocals".

        mode ({"vanilla-diffrhythm", "instrumental-to-vocals", "instrumental-condition"}):
            Selects sampling behavior:
            • "vanilla-diffrhythm": standard DiffRhythm generation/inpainting.
            • "instrumental-to-vocals": Start the diffusion process at x0
            • "instrumental-condition": uses instrumental latents as conditioning but
              does not mix them back into the output.
    """
        self.eval()

        if next(self.parameters()).dtype == torch.float16:
            cond = cond.half()

        if cond.shape[1] > duration:
            cond = cond[:, :duration, :]

        # raw wave
        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # text
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # duration and conditioning mask
        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        # default: full-sequence prediction if no segments are given
        if latent_pred_segments is None:
            latent_pred_segments = [[0, duration]] * batch

        latent_pred_segments = torch.tensor(latent_pred_segments, device=cond.device)
        fixed_span_mask = custom_mask_from_start_end_indices(
            cond_seq_len,
            latent_pred_segments,
            device=cond.device,
            max_seq_len=duration
        ).unsqueeze(-1)

        if mode == "instrumental-condition":
            step_cond = cond
        else:
            # Zero out conditioning inside editable spans
            step_cond = torch.where(fixed_span_mask, torch.zeros_like(cond), cond)

        if isinstance(duration, int):
            duration = torch.full((batch_infer_num,), duration, device=device, dtype=torch.long)

        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(
                cond,
                (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len),
                value=0.0
            )

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # optional: remove reference audio conditioning completely
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        # repeats become no-ops if batch_infer_num = 1
        cond = cond.repeat(batch_infer_num, 1, 1)
        step_cond = step_cond.repeat(batch_infer_num, 1, 1)
        text = text.repeat(batch_infer_num, 1)
        style_prompt = style_prompt.repeat(batch_infer_num, 1)
        negative_style_prompt = negative_style_prompt.repeat(batch_infer_num, 1)
        start_time = start_time.repeat(batch_infer_num)
        fixed_span_mask = fixed_span_mask.repeat(batch_infer_num, 1, 1)
        song_duration = song_duration.repeat(batch_infer_num)

        def fn(t, x):
            # predict flow
            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, drop_audio_cond=False, drop_text=False, drop_prompt=False,
                style_prompt=style_prompt, start_time=start_time, duration=song_duration
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, drop_audio_cond=True, drop_text=True, drop_prompt=False,
                style_prompt=negative_style_prompt, start_time=start_time, duration=song_duration
            )
            return pred + (pred - null_pred) * cfg_strength

        # prepare initial state y0
        if mode == "instrumental-to-vocals":
            # start sampling from provided x0
            assert x0 is not None, "x0 must be provided for instrumental-to-vocals mode"
            y0 = x0
        else:
            # start from Gaussian noise
            y0 = []
            for dur in duration:
                if exists(seed):
                    torch.manual_seed(seed)
                y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
            y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))
        
        t = torch.linspace(t_start, 1, steps, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)

        sampled = trajectory[-1]
        out = sampled

        if mode == "vanilla-diffrhythm":
            out = torch.where(fixed_span_mask, out, cond)
        else:
            # Do not copy condition back to output if mode is
            # "instrumental-to-vocals" or "instrumental-condition"
            pass

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        out = torch.chunk(out, batch_infer_num, dim=0)
        return out, trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        style_prompt = None,
        lens: int["b"] | None = None,  # noqa: F821
        noise_scheduler: str | None = None,
        grad_ckpt = False,
        start_time = None,
        x0: float["b n d"] | None = None,  # noqa: F722
        cond: float["b n d"] | None = None,  # noqa: F722
        mode: Literal["vanilla-diffrhythm", "instrumental-to-vocals", "instrumental-condition"] = "vanilla-diffrhythm",
        **kwargs
    ):

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)

        # True for valid frames, False for padding
        padding_mask = lens_to_mask(lens, length=seq_len)  # useless here, as collate_fn will pad to max length in batch

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths, self.max_frames)

        if exists(padding_mask):
            # NOTE: this is weird; rand_span_mask is never used and instead overwritten with padding_mask
            # For in-filling task we would want to use rand_span_mask to zero out a random span
            # Also see: Below they zero out everything outside of the valid lengths meaning nothing of the condition remains!
            rand_span_mask = padding_mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        if x0 is None:
            # Use standard Gaussian noise as x0
            x0 = torch.randn_like(x1)
        else:
            # Use the provided x0
            assert x0.shape == x1.shape, "Provided x0 and inp must have the same shape"
            x0 = x0
            
        # time step
        time = torch.normal(mean=0, std=1, size=(batch,), device=self.device)
        time = torch.nn.functional.sigmoid(time)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        if cond is None:
            # NOTE: weird thing continues; note that the values for input and other should be changed
            # For in-filling we would want to randomly show parts of the ground truth, i.e. x1/inp
            # this is the other way around but works since they drop the in-filling task and mask out wherever there are valid frames
            # Their original comment: "Create condition by masking out random spans of the target" is misleading!
            cond = torch.where(rand_span_mask[..., None], input=torch.zeros_like(x1), other=x1)
        else:
            # Show full condition, but update mask so that loss is computed over everything except padding
            cond = cond
            rand_span_mask = padding_mask

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        drop_text = random() < self.lrc_drop_prob
        drop_prompt = random() < self.style_drop_prob

        # if want rigourously mask out padding, record in collate_fn in dataset.py, and pass in here
        # adding mask will use more memory, thus also need to adjust batchsampler with scaled down threshold for long sequences
        pred = self.transformer(
            x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text, drop_prompt=drop_prompt,
            style_prompt=style_prompt, start_time=start_time
        )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred
""