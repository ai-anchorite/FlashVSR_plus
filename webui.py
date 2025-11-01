import sys
import argparse
import gradio as gr
import os
import re
import math
import uuid
import torch
import shutil
import imageio
import ffmpeg
import numpy as np
import torch.nn.functional as F
import random
import time
import subprocess
import psutil
import tempfile
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from huggingface_hub import snapshot_download
from gradio_videoslider import VideoSlider

from src import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline
from src.models import wan_video_dit
from src.models.TCDecoder import build_tcdecoder
from src.models.utils import get_device_list, clean_vram, Buffer_LQ4x_Proj

from toolbox.system_monitor import SystemMonitor
from toolbox.toolbox import ToolboxProcessor

# Initialize toolbox_processor after load_config is defined
toolbox_processor = None

# Suppress annoyingly persistent Windows asyncio proactor errors
if os.name == 'nt':  # Windows only
    import asyncio
    from functools import wraps
    import socket # Required for the ConnectionResetError
    
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    def silence_connection_errors(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except (ConnectionResetError, BrokenPipeError):
                pass
            except RuntimeError as e:
                if str(e) != 'Event loop is closed':
                    raise
        return wrapper
    
    from asyncio import proactor_events
    if hasattr(proactor_events, '_ProactorBasePipeTransport'):
        proactor_events._ProactorBasePipeTransport._call_connection_lost = silence_connection_errors(
            proactor_events._ProactorBasePipeTransport._call_connection_lost
        )

parser = argparse.ArgumentParser(description="FlashVSR+ WebUI")
parser.add_argument("--listen", action="store_true", help="Allow LAN access")
parser.add_argument("--port", type=int, default=7860, help="Service Port")
args = parser.parse_args()
        
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
TEMP_DIR = os.path.join(ROOT_DIR, "_temp")
CONFIG_FILE = os.path.join(ROOT_DIR, "webui_config")
os.environ['GRADIO_TEMP_DIR'] = TEMP_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

def load_config():
    """Load user preferences from config file."""
    config = {"clear_temp_on_start": False, "autosave": True, "tb_autosave": True}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                for line in f:
                    key, value = line.strip().split('=')
                    config[key] = value.lower() == 'true'
        except:
            pass
    return config

def save_config(config):
    """Save user preferences to config file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            for key, value in config.items():
                f.write(f"{key}={value}\n")
    except Exception as e:
        log(f"Error saving config: {e}", message_type="error")

def log(message:str, message_type:str="normal"):
    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    elif message_type == 'info':
        message = '\033[1;33m' + message + '\033[m'
    else:
        message = message
    print(f"{message}")

def dummy_tqdm(iterable, *args, **kwargs):
    return iterable

def model_download(model_name="JunhaoZhuang/FlashVSR"):
    model_dir = os.path.join(ROOT_DIR, "models", "FlashVSR")
    if not os.path.exists(model_dir):
        log(f"Downloading model '{model_name}' from huggingface...", message_type='info')
        snapshot_download(repo_id=model_name, local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)
        log("Model download complete!", message_type='finish')
        print()

def tensor2video(frames: torch.Tensor):
    video_squeezed = frames.squeeze(0)
    video_permuted = rearrange(video_squeezed, "C F H W -> F H W C")
    video_final = (video_permuted.float() + 1.0) / 2.0
    return video_final

def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', os.path.basename(name))]

def list_images_natural(folder: str):
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs

def largest_8n1_leq(n):
    # Find largest value of form 8n+1 that is <= n
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def smallest_8n1_geq(n):
    # Find smallest value of form 8n+1 that is >= n (rounds up to preserve frames)
    if n < 1:
        return 1
    # If n is already 8k+1, return n
    if (n - 1) % 8 == 0:
        return n
    # Otherwise round up
    return ((n - 1)//8 + 1)*8 + 1

def is_video(path):
    return os.path.isfile(path) and path.lower().endswith(('.mp4','.mov','.avi','.mkv'))

def is_ffmpeg_available():
    return shutil.which("ffmpeg") is not None

def save_video(frames, save_path, fps=30, quality=5, progress_desc="Saving video..."):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    frames_np = (frames.cpu().float() * 255.0).clip(0, 255).numpy().astype(np.uint8)
    with imageio.get_writer(save_path, fps=fps, quality=quality, macro_block_size=1) as writer:
        for frame_np in tqdm(frames_np, desc=f"[FlashVSR] {progress_desc}"):
            writer.append_data(frame_np)

def prepare_tensors(path: str, dtype=torch.bfloat16):
    if os.path.isdir(path):
        paths0 = list_images_natural(path)
        if not paths0: raise FileNotFoundError(f"No images in {path}")
        with Image.open(paths0[0]) as _img0: w0, h0 = _img0.size
        frames = [torch.from_numpy(np.array(Image.open(p).convert('RGB')).astype(np.float32) / 255.0).to(dtype) for p in tqdm(paths0, desc="Loading images")]
        return torch.stack(frames, 0), 30
    if is_video(path):
        with imageio.get_reader(path) as rdr:
            meta = rdr.get_meta_data()
            fps = meta.get('fps', 30)
            frames = [torch.from_numpy(frame_data.astype(np.float32) / 255.0).to(dtype) for frame_data in tqdm(rdr, desc="Loading video frames")]
        return torch.stack(frames, 0), fps
    raise ValueError(f"Unsupported input: {path}")

def get_input_params(image_tensor, scale):
    N0, h0, w0, _ = image_tensor.shape
    multiple = 128
    sW, sH, tW, tH = w0 * scale, h0 * scale, max(multiple, (w0 * scale // multiple) * multiple), max(multiple, (h0 * scale // multiple) * multiple)
    # Use smallest_8n1_geq to round UP and preserve all frames
    F = smallest_8n1_geq(N0 + 4)
    if F == 0: raise RuntimeError(f"Not enough frames. Got {N0 + 4}.")
    return tH, tW, F

def input_tensor_generator(image_tensor: torch.Tensor, device, scale: int = 4, dtype=torch.bfloat16):
    N0, h0, w0, _ = image_tensor.shape
    tH, tW, Fs = get_input_params(image_tensor, scale)
    for i in range(Fs):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor_bchw = frame_slice.permute(2, 0, 1).unsqueeze(0)
        upscaled_tensor = F.interpolate(tensor_bchw, size=(h0 * scale, w0 * scale), mode='bicubic', align_corners=False)
        l, t = max(0, (w0 * scale - tW) // 2), max(0, (h0 * scale - tH) // 2)
        cropped_tensor = upscaled_tensor[:, :, t:t + tH, l:l + tW]
        tensor_out = (cropped_tensor.squeeze(0) * 2.0 - 1.0)
        yield tensor_out.to('cpu').to(dtype)

def prepare_input_tensor(image_tensor: torch.Tensor, device, scale: int = 4, dtype=torch.bfloat16):
    N0, h0, w0, _ = image_tensor.shape
    tH, tW, Fs = get_input_params(image_tensor, scale)
    frames = []
    for i in range(Fs):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor_bchw = frame_slice.permute(2, 0, 1).unsqueeze(0)
        upscaled_tensor = F.interpolate(tensor_bchw, size=(h0 * scale, w0 * scale), mode='bicubic', align_corners=False)
        l, t = max(0, (w0 * scale - tW) // 2), max(0, (h0 * scale - tH) // 2)
        cropped_tensor = upscaled_tensor[:, :, t:t + tH, l:l + tW]
        tensor_out = (cropped_tensor.squeeze(0) * 2.0 - 1.0).to('cpu').to(dtype)
        frames.append(tensor_out)
    vid_stacked = torch.stack(frames, 0)
    vid_final = vid_stacked.permute(1, 0, 2, 3).unsqueeze(0)
    clean_vram()
    return vid_final, tH, tW, Fs

def calculate_tile_coords(height, width, tile_size, overlap):
    coords = []
    stride = tile_size - overlap
    num_rows, num_cols = math.ceil((height - overlap) / stride), math.ceil((width - overlap) / stride)
    for r in range(num_rows):
        for c in range(num_cols):
            y1, x1 = r * stride, c * stride
            y2, x2 = min(y1 + tile_size, height), min(x1 + tile_size, width)
            if y2 - y1 < tile_size: y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size: x1 = max(0, x2 - tile_size)
            coords.append((x1, y1, x2, y2))
    return coords

def create_feather_mask(size, overlap):
    H, W = size
    mask = torch.ones(1, 1, H, W)
    ramp = torch.linspace(0, 1, overlap)
    mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp.view(1, 1, 1, -1))
    mask[:, :, :, -overlap:] = torch.minimum(mask[:, :, :, -overlap:], ramp.flip(0).view(1, 1, 1, -1))
    mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
    mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1))
    return mask

def stitch_video_tiles(
    tile_paths,
    tile_coords,
    final_dims,
    scale,
    overlap,
    output_path,
    fps,
    quality,
    cleanup=True,
    chunk_size=40
):
    if not tile_paths:
        log("No tile videos found to stitch.", message_type='error')
        return

    final_W, final_H = final_dims

    readers = [imageio.get_reader(p) for p in tile_paths]

    try:
        num_frames = readers[0].count_frames()
        if num_frames is None or num_frames <= 0:
            num_frames = len([_ for _ in readers[0]])
            for r in readers: r.close()
            readers = [imageio.get_reader(p) for p in tile_paths]

        with imageio.get_writer(output_path, fps=fps, quality=quality, macro_block_size=1) as writer:
            for start_frame in tqdm(range(0, num_frames, chunk_size), desc="[FlashVSR] Stitching Chunks"):
                end_frame = min(start_frame + chunk_size, num_frames)
                current_chunk_size = end_frame - start_frame
                chunk_canvas = np.zeros((current_chunk_size, final_H, final_W, 3), dtype=np.float32)
                weight_canvas = np.zeros_like(chunk_canvas, dtype=np.float32)

                for i, reader in enumerate(readers):
                    try:
                        tile_chunk_frames = [
                            frame.astype(np.float32) / 255.0
                            for idx, frame in enumerate(reader.iter_data())
                            if start_frame <= idx < end_frame
                        ]
                        tile_chunk_np = np.stack(tile_chunk_frames, axis=0)
                    except Exception as e:
                        log(f"Warning: Could not read chunk from tile {i}. Error: {e}", message_type='warning')
                        continue

                    if tile_chunk_np.shape[0] != current_chunk_size:
                        log(f"Warning: Tile {i} chunk has incorrect frame count. Skipping.", message_type='warning')
                        continue

                    tile_H, tile_W, _ = tile_chunk_np.shape[1:]
                    ramp = np.linspace(0, 1, overlap * scale, dtype=np.float32)
                    mask = np.ones((tile_H, tile_W, 1), dtype=np.float32)
                    mask[:, :overlap*scale, :] *= ramp[np.newaxis, :, np.newaxis]
                    mask[:, -overlap*scale:, :] *= np.flip(ramp)[np.newaxis, :, np.newaxis]
                    mask[:overlap*scale, :, :] *= ramp[:, np.newaxis, np.newaxis]
                    mask[-overlap*scale:, :, :] *= np.flip(ramp)[:, np.newaxis, np.newaxis]
                    mask_4d = mask[np.newaxis, :, :, :]

                    x1_orig, y1_orig, _, _ = tile_coords[i]
                    out_y1, out_x1 = y1_orig * scale, x1_orig * scale
                    out_y2, out_x2 = out_y1 + tile_H, out_x1 + tile_W

                    chunk_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += tile_chunk_np * mask_4d
                    weight_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_4d

                weight_canvas[weight_canvas == 0] = 1.0
                stitched_chunk = chunk_canvas / weight_canvas

                for frame_idx_in_chunk in range(current_chunk_size):
                    frame_uint8 = (np.clip(stitched_chunk[frame_idx_in_chunk], 0, 1) * 255).astype(np.uint8)
                    writer.append_data(frame_uint8)

    finally:
        log("Closing all tile reader instances...")
        for reader in readers:
            reader.close()

    if cleanup:
        log("Cleaning up temporary tile files...")
        for path in tile_paths:
            try:
                os.remove(path)
            except OSError as e:
                log(f"Could not remove temporary file '{path}': {e}", message_type='warning')


def merge_video_with_audio(video_only_path, audio_source_path, output_path):
    """
    Merges the video from video_only_path with audio from audio_source_path into output_path.
    Provides clean, concise logging and gracefully handles errors.
    """
    if not is_ffmpeg_available():
        shutil.move(video_only_path, output_path)
        log("[FlashVSR] FFmpeg not found. The video has been processed without audio.", message_type='warning')
        return

    try:
        # Check if the source video has an audio stream
        probe = ffmpeg.probe(audio_source_path)
        if not any(s['codec_type'] == 'audio' for s in probe.get('streams', [])):
            shutil.move(video_only_path, output_path)
            log("[FlashVSR] No audio stream found in the source. The video has been processed without audio.", message_type='info')
            return
    except ffmpeg.Error:
        # If probing fails, we can't get the audio.
        shutil.move(video_only_path, output_path)
        log("[FlashVSR] Could not probe source for audio. The video has been processed without audio.", message_type='warning')
        return

    try:
        # Perform the merge
        input_video = ffmpeg.input(video_only_path)
        input_audio = ffmpeg.input(audio_source_path)
        ffmpeg.output(
            input_video['v'],
            input_audio['a'],
            output_path,
            vcodec='copy',
            acodec='copy'
        ).run(overwrite_output=True, quiet=True)

        log("[FlashVSR] Audio successfully merged.", message_type='finish')

    except ffmpeg.Error:
        # If the merge operation fails, save the silent video.
        shutil.move(video_only_path, output_path)
        log("[FlashVSR] Audio merge failed. The video has been processed without audio.", message_type='warning')

    finally:
        # Clean up the source video-only file if it still exists
        if os.path.exists(video_only_path):
            try:
                os.remove(video_only_path)
            except OSError as e:
                log(f"[FlashVSR] Could not remove temporary file '{video_only_path}': {e}", message_type='error')

def save_file_manually(temp_path):
    if not temp_path or not os.path.exists(temp_path):
        log("Error: No file to save.", message_type="error")
        return '<div style="padding: 1px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 1px; color: #721c24;">❌ No file to save.</div>'
    
    filename = os.path.basename(temp_path)
    final_path = os.path.join(OUTPUT_DIR, filename)
    
    try:
        shutil.copy(temp_path, final_path)
        log(f"File saved to: {final_path}", message_type="finish")
        return f'<div style="padding: 1px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 1px; color: #155724;">✅ File saved to: {final_path}</div>'
    except Exception as e:
        log(f"Error saving file: {e}", message_type="error")
        return f'<div style="padding: 1px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 1px; color: #721c24;">❌ Error saving file: {e}</div>'

def clear_temp_files():
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            os.makedirs(TEMP_DIR, exist_ok=True)
            log("Temp files cleared.", message_type="finish")
            return '<div style="padding: 1px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 1px; color: #155724;">✅ Temp files cleared.</div>'
        else:
            log("Temp directory doesn't exist.", message_type="info")
            return '<div style="padding: 1px; background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 1px; color: #0c5460;">ℹ️ Temp directory doesn\'t exist.</div>'
    except Exception as e:
        log(f"Error clearing temp files: {e}", message_type="error")
        return f'<div style="padding: 1px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 1px; color: #721c24;">❌ Error clearing temp files: {e}</div>'
    

def init_pipeline(mode, device, dtype):
    model_download()
    model_path = os.path.join(ROOT_DIR, "models", "FlashVSR")
    ckpt_path, vae_path, lq_path, tcd_path, prompt_path = [os.path.join(model_path, f) for f in ["diffusion_pytorch_model_streaming_dmd.safetensors", "Wan2.1_VAE.pth", "LQ_proj_in.ckpt", "TCDecoder.ckpt", "../posi_prompt.pth"]]
    mm = ModelManager(torch_dtype=dtype, device="cpu")
    if mode == "full":
        mm.load_models([ckpt_path, vae_path]); pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
    else:
        mm.load_models([ckpt_path]); pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device) if mode == "tiny" else FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
        pipe.TCDecoder = build_tcdecoder(new_channels=[512, 256, 128, 128], device=device, dtype=dtype, new_latent_channels=16+768)
        pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location=device, weights_only=False), strict=False); pipe.TCDecoder.clean_mem()
    pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    if os.path.exists(lq_path): pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_path, map_location="cpu", weights_only=False), strict=True)
    pipe.to(device, dtype=dtype); pipe.enable_vram_management(); pipe.init_cross_kv(prompt_path=prompt_path); pipe.load_models_to_device(["dit", "vae"])
    return pipe

# --- Integrated Core Logic Function (Updated) ---
def run_flashvsr_integrated(
    input_path,
    mode,
    scale,
    color_fix,
    tiled_vae,
    tiled_dit,
    tile_size,
    tile_overlap,
    unload_dit,
    dtype_str,
    seed,
    device,
    fps_override,
    quality,
    attention_mode,
    sparse_ratio,
    kv_ratio,
    local_range,
    autosave,
    progress=gr.Progress(track_tqdm=True)
):
    if not input_path: raise gr.Error("Please provide an input video or image folder path!")
    if seed == -1: seed = random.randint(0, 2**32 - 1)

    # --- Parameter Preparation ---
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}; dtype = dtype_map.get(dtype_str, torch.bfloat16)
    devices = get_device_list(); _device = device
    if device == "auto": _device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    if _device not in devices and _device != "cpu": raise gr.Error(f"Device '{_device}' is not available! Available devices: {devices}")
    if _device.startswith("cuda"): torch.cuda.set_device(_device)
    if tiled_dit and (tile_overlap > tile_size / 2): raise gr.Error("The overlap must be less than half of the tile size!")
    wan_video_dit.USE_BLOCK_ATTN = (attention_mode == "block")

    # --- Output Path ---
    input_basename = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_filename = f"{input_basename}_{mode}_s{scale}_{timestamp}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    temp_video_path = os.path.join(TEMP_DIR, f"video_only_{output_filename}")
    final_output_location = os.path.join(OUTPUT_DIR, output_filename) if autosave else os.path.join(TEMP_DIR, output_filename)


    # --- Core Logic ---
    progress(0, desc="Loading video frames...")
    log(f"Loading frames from {input_path}...", message_type='info')
    frames, original_fps = prepare_tensors(input_path, dtype=dtype)
    _fps = original_fps if is_video(input_path) else fps_override
    if frames.shape[0] < 21: raise gr.Error(f"Input must have at least 21 frames, but got {frames.shape[0]} frames.")
    log("Video frames loaded successfully.", message_type="finish")

    final_output_tensor = None

    # Build a common pipe parameter dictionary
    pipe_kwargs = {
        "prompt": "", "negative_prompt": "", "cfg_scale": 1.0, "num_inference_steps": 1,
        "seed": seed, "tiled": tiled_vae, "is_full_block": False, "if_buffer": True,
        "kv_ratio": kv_ratio, "local_range": local_range, "color_fix": color_fix,
        "unload_dit": unload_dit, "fps": _fps, "tiled_dit": tiled_dit,
    }

    if tiled_dit:
        N, H, W, C = frames.shape
        progress(0.1, desc="Initializing model pipeline...")
        pipe = init_pipeline(mode, _device, dtype)
        tile_coords = calculate_tile_coords(H, W, tile_size, tile_overlap)

        if mode == "tiny-long":
            local_temp_dir = os.path.join(TEMP_DIR, str(uuid.uuid4())); os.makedirs(local_temp_dir, exist_ok=True)
            temp_videos = []
            for i in tqdm(range(len(tile_coords)), desc="[FlashVSR] Processing tiles"):
                x1, y1, x2, y2 = tile_coords[i]
                input_tile = frames[:, y1:y2, x1:x2, :]
                temp_name = os.path.join(local_temp_dir, f"{i+1:05d}.mp4")
                th, tw, F = get_input_params(input_tile, scale)
                LQ_tile = input_tensor_generator(input_tile, _device, scale=scale, dtype=dtype)
                pipe(
                    LQ_video=LQ_tile, num_frames=F, height=th, width=tw,
                    topk_ratio=sparse_ratio*768*1280/(th*tw),
                    quality=10, output_path=temp_name, **pipe_kwargs
                )
                temp_videos.append(temp_name); del LQ_tile, input_tile; clean_vram()

            stitch_video_tiles(temp_videos, tile_coords, (W*scale, H*scale), scale, tile_overlap, temp_video_path, _fps, quality, True)
            shutil.rmtree(local_temp_dir)
        else: # Stitch in memory
            # Output should match input frame count - model adds context internally
            num_aligned_frames = N
            final_output_canvas, weight_sum_canvas = torch.zeros((num_aligned_frames, H*scale, W*scale, C), dtype=torch.float32), torch.zeros((num_aligned_frames, H*scale, W*scale, C), dtype=torch.float32)
            for i in tqdm(range(len(tile_coords)), desc="[FlashVSR] Processing tiles"):
                x1, y1, x2, y2 = tile_coords[i]
                input_tile = frames[:, y1:y2, x1:x2, :]
                LQ_tile, th, tw, F = prepare_input_tensor(input_tile, _device, scale=scale, dtype=dtype)
                LQ_tile = LQ_tile.to(_device)
                output_tile_gpu = pipe(
                    LQ_video=LQ_tile, num_frames=F, height=th, width=tw,
                    topk_ratio=sparse_ratio*768*1280/(th*tw), **pipe_kwargs
                )
                processed_tile_cpu = tensor2video(output_tile_gpu).cpu()
                # Trim to match input frame count if model output more frames
                processed_tile_cpu = processed_tile_cpu[:num_aligned_frames]
                mask = create_feather_mask((processed_tile_cpu.shape[1], processed_tile_cpu.shape[2]), tile_overlap * scale).cpu().permute(0, 2, 3, 1)
                x1_s, y1_s = x1 * scale, y1 * scale
                x2_s, y2_s = x1_s + processed_tile_cpu.shape[2], y1_s + processed_tile_cpu.shape[1]
                final_output_canvas[:, y1_s:y2_s, x1_s:x2_s, :] += processed_tile_cpu * mask
                weight_sum_canvas[:, y1_s:y2_s, x1_s:x2_s, :] += mask
                del LQ_tile, output_tile_gpu, processed_tile_cpu, input_tile; clean_vram()
            weight_sum_canvas[weight_sum_canvas == 0] = 1.0
            final_output_tensor = final_output_canvas / weight_sum_canvas
    else: # Non-tiled mode
        progress(0.1, desc="Initializing model pipeline...")
        pipe = init_pipeline(mode, _device, dtype)
        log(f"Processing {frames.shape[0]} frames...", message_type='info')

        th, tw, F = get_input_params(frames, scale)
        if mode == "tiny-long":
            LQ = input_tensor_generator(frames, _device, scale=scale, dtype=dtype)
            pipe(
                LQ_video=LQ, num_frames=F, height=th, width=tw,
                topk_ratio=sparse_ratio*768*1280/(th*tw),
                output_path=temp_video_path, quality=quality, **pipe_kwargs
            )
        else:
            LQ, _, _, _ = prepare_input_tensor(frames, _device, scale=scale, dtype=dtype)
            LQ = LQ.to(_device)
            video = pipe(
                LQ_video=LQ, num_frames=F, height=th, width=tw,
                topk_ratio=sparse_ratio*768*1280/(th*tw), **pipe_kwargs
            )
            final_output_tensor = tensor2video(video).cpu()
            # Trim to match input frame count
            final_output_tensor = final_output_tensor[:frames.shape[0]]
        del pipe; clean_vram()

    if final_output_tensor is not None:
        progress(0.9, desc="Saving final video...")
        save_video(final_output_tensor, temp_video_path, fps=_fps, quality=quality)

    # Always save to temp directory first (persists during session)
    temp_output_path = os.path.join(TEMP_DIR, output_filename)

    if is_video(input_path):
        progress(0.95, desc="Merging audio...")
        merge_video_with_audio(temp_video_path, input_path, temp_output_path)
    else:
        shutil.move(temp_video_path, temp_output_path)
    
    # Autosave to outputs folder if enabled
    if autosave:  
        final_save_path = os.path.join(OUTPUT_DIR, output_filename)
        shutil.copy(temp_output_path, final_save_path)
        log(f"Processing complete! Auto-saved to: {final_save_path}", message_type="finish")
    else:
        log(f"Processing complete! Use 'Save Output' to save to outputs folder.", message_type="finish")
    
    progress(1, desc="Done!")
    
    # Return: video_output, output_file_path, video_slider_output
    return (
        temp_output_path,
        temp_output_path,
        (input_path, temp_output_path)
    ) 


def open_folder(folder_path):
    try:
        if sys.platform == "win32":
            os.startfile(folder_path)
        elif sys.platform == "darwin":
            subprocess.run(["open", folder_path])
        else:
            subprocess.run(["xdg-open", folder_path])
        return f'<div style="padding: 1px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 1px; color: #155724;">✅ Opened folder: {folder_path}</div>'
    except Exception as e:
        return f'<div style="padding: 1px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 4px; color: #721c24;">❌ Error opening folder: {e}</div>'

def save_file(file_path):
    if file_path and os.path.exists(file_path):
        log(f"File saved to: {file_path}", message_type="finish")
    else:
        log(f"File not found or unable to save.", message_type="error")

def handle_start_pipeline(
    active_tab_index, single_video_path, batch_video_paths, selected_ops,
    # Frame Adjust params
    fps_mode, speed_factor, frames_use_streaming, frames_quality,
    # Video Loop params
    loop_type, num_loops,
    # Export params
    export_format, quality, max_width, output_name,
    progress=gr.Progress()
):
    # Determine input paths based on the active tab
    if active_tab_index == 1 and batch_video_paths:
        input_paths = [file.name for file in batch_video_paths]
        if not input_paths:
            return None, "⚠️ Batch Input tab is active, but no files were provided."
    elif active_tab_index == 0 and single_video_path:
        input_paths = [single_video_path]
    else:
        return None, "⚠️ No input video found in the active tab. Please upload a video."

    if not selected_ops:
        return None, "⚠️ No operations selected. Please check at least one box in 'Pipeline Steps'."

    # Pack parameters for the processor
    params = {
        "frame_adjust": {
            "fps_mode": fps_mode, "speed_factor": speed_factor, "use_streaming": frames_use_streaming, "output_quality": frames_quality
        },
        "loop": {
            "loop_type": loop_type, "num_loops": num_loops
        },
        "export": {
            "export_format": export_format, "quality": quality, "max_width": max_width, "output_name": output_name
        }
    }
    
    if len(input_paths) > 1:
        # Batch processing
        final_video, message = toolbox_processor.process_batch(input_paths, selected_ops, params, progress)
    else:
        # Single video processing
        temp_video, message = toolbox_processor.process_pipeline(input_paths[0], selected_ops, params, progress)
        final_video = None
        if temp_video:
            if toolbox_processor.autosave_enabled:
                temp_path = Path(temp_video)
                final_path = toolbox_processor.output_dir / temp_path.name
                final_video = toolbox_processor._copy_to_permanent_storage(temp_video, final_path)
                message += f"\n✅ Autosaved result to: {final_path}"
            else:
                final_video = temp_video # Leave in temp folder for manual save
                message += "\nℹ️ Autosave is off. Result is in a temporary folder. Use 'Manual Save' to keep it."

    return final_video, message
    
# Idle state HTML options for save_status display (compact versions)
IDLE_STATES = [
    # Option 1: Compact Gradient
    '''<div style="padding: 1px; text-align: center;">
        <span style="
            font-size: 1.1em;
            font-weight: 600;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        ">FlashVSR+</span>
    </div>'''
]

css = """
.video-window {
    min-height: 300px !important;
    height: auto !important;
}

.video-window video, .image-window img {
    max-height: 60vh !important;
    object-fit: contain;
    width: 100%;
}
.video-window .source-selection {
    display: none !important;
}

/* Compact monitor styling */
.monitor-compact textarea {
    min-width: 0 !important;
    font-size: 0.9em !important;
    padding: 2px !important;
}

.monitor-compact {
    min-width: 0 !important;
}
"""
    
def create_ui():
    global toolbox_processor
    
    # Initialize toolbox processor with shared config
    if toolbox_processor is None:
        config = load_config()
        toolbox_processor = ToolboxProcessor(config.get("tb_autosave", True))
    
    with gr.Blocks(css=css) as demo:
        output_file_path = gr.State(None)

        with gr.Tabs(elem_id="main_tabs") as main_tabs:
            with gr.TabItem("FlashVSR", id=0):
                with gr.Row():
                    # --- Left-side Column ---                       
                    with gr.Column(scale=1):
                        input_video = gr.Video(label="Upload Video File", elem_classes="video-window")
                        run_button = gr.Button("Start Processing", variant="primary", size="sm")
                        with gr.Group():
                            with gr.Row():
                                mode_radio = gr.Radio(choices=["tiny", "full"], value="tiny", label="Pipeline Mode", info="'Full' requires 24GB+ VRAM")
                                seed_number = gr.Number(value=-1, label="Seed", precision=0, info="-1 = random")
                        with gr.Group():
                            with gr.Row():
                                scale_slider = gr.Slider(minimum=2, maximum=4, step=1, value=2, label="Upscale Factor", info="Designed to upscale small/short AI video. Start with x2...")
                                tiled_dit_checkbox = gr.Checkbox(label="Enable Tiled DiT", info="Greatly reduces VRAM at the cost of speed.", value=True)
                            with gr.Row(visible=False) as tiled_dit_options:
                                tile_size_slider = gr.Slider(minimum=64, maximum=512, step=16, value=256, label="Tile Size")
                                tile_overlap_slider = gr.Slider(minimum=8, maximum=128, step=8, value=24, label="Tile Overlap")
                                
                    # --- Right-side Column ---      
                    with gr.Column(scale=1):
                        video_output = gr.Video(label="Output Result", interactive=False, elem_classes="video-window")
                        with gr.Group():
                            with gr.Row():                            
                                save_button = gr.Button("Save Manually 💾", size="sm", variant="primary")
                                send_to_toolbox_btn = gr.Button("Send to Toolbox 🛠️", size="sm")                            
                            with gr.Row():
                                config = load_config()
                                autosave_checkbox = gr.Checkbox(label="Autosave Output", value=config.get("autosave", True))
                                clear_on_start_checkbox = gr.Checkbox(label="Clear Temp on Start", value=config.get("clear_temp_on_start", False))
                            with gr.Row():                                
                                open_folder_button = gr.Button("Open Output Folder", size="sm")
                                clear_temp_button = gr.Button("Clear Temp Files", size="sm", variant="secondary")
                        with gr.Row():
                            save_status = gr.HTML(
                                value=random.choice(IDLE_STATES),
                                padding=False
                            )                         
                        with gr.Row():
                            with gr.Column(scale=1, min_width=200):
                                gpu_monitor = gr.Textbox(
                                    lines=4,
                                    container=False,
                                    interactive=False,
                                    elem_classes="monitor-compact"
                                )
                            with gr.Column(scale=1, min_width=200):
                                cpu_monitor = gr.Textbox(
                                    lines=2,
                                    container=False,
                                    interactive=False,
                                    elem_classes="monitor-compact"
                                )
                                
                # --- Advanced Options ---  
                with gr.Row():
                    with gr.Accordion("Advanced Options", open=False):
                        sparse_ratio_slider = gr.Slider(minimum=0.5, maximum=5.0, step=0.1, value=2.0, label="Sparse Ratio", info="Recommended: 1.5 or 2.0. 1.5 → faster; 2.0 → more stable")
                        kv_ratio_slider = gr.Slider(minimum=1, maximum=8, step=1, value=3, label="KV Ratio", info="Controls the length of the KV cache")
                        local_range_slider = gr.Slider(minimum=3, maximum=15, step=2, value=11, label="Local Range", info="Recommended: 9 or 11. 9 → sharper details; 11 → more stable results.")
                        attention_mode_radio = gr.Radio(choices=["sage", "block"], value="sage", label="Attention Mode")
                        color_fix_checkbox = gr.Checkbox(label="Enable Color Fix", value=True)
                        tiled_vae_checkbox = gr.Checkbox(label="Enable Tiled VAE", value=True)
                        unload_dit_checkbox = gr.Checkbox(label="Unload DiT Before Decoding (Saves VRAM)", value=False)
                        dtype_radio = gr.Radio(choices=["fp16", "bf16"], value="bf16", label="Data Type")
                        device_textbox = gr.Textbox(value="auto", label="Device", info="E.g.: 'auto', 'cuda:0', 'cpu'")
                        quality_slider = gr.Slider(minimum=1, maximum=10, step=1, value=8, label="Output Video Quality")
                        fps_number = gr.Number(value=30, label="Output FPS (Only for image sequence input)", precision=0)

                # --- Main Tab's VideoSlider output ---  
                with gr.Row():
                    video_slider_output = VideoSlider(
                        label="Video Comparison",
                        interactive=False,
                        video_mode="preview",
                        show_download_button=False,
                        autoplay=True, 
                        loop=True,
                        height=800,
                        width=1200
                    )  
            
            # --- TOOLBOX TAB ---
            with gr.TabItem("🛠️ Toolbox", id=1):
                with gr.Row():
                    # --- Left Column: Inputs and Pipeline Control ---
                    with gr.Column(scale=1):
                        # Hidden state to track the active input tab (0=Single, 1=Batch)
                        tb_active_tab_index = gr.Number(value=0, visible=False)
                        
                        with gr.Tabs() as tb_input_tabs:
                            with gr.TabItem("Single Video", id=0):
                                 tb_input_video = gr.Video(label="Toolbox Input Video", autoplay=True, elem_classes="video-window")
                            with gr.TabItem("Batch Video", id=1):
                                tb_batch_input_files = gr.File(
                                    label="Upload Multiple Videos for Batch Processing",
                                    file_count="multiple",
                                    type="filepath",
                                    height="300px",                            
                                )
                            tb_start_pipeline_btn = gr.Button("🚀 Start Pipeline Processing", variant="primary", size="sm")                              
                            with gr.Group():
                                tb_pipeline_steps_chkbox = gr.CheckboxGroup(
                                    choices=["Frame Adjust", "Video Loop", "Export"],
                                    value=[],
                                    show_label=False,
                                    info="Preconfigure the Operations Settings in the section below and use these checkboxes to run them in order. The 'Export' option is primarily for reducing the video filesize for posting to social media, etc. Note that batch processing requires at least one checkbox checked."
                                )
                            
                            # Video Analysis Section
                            tb_analyze_button = gr.Button("📊 Analyze Input Video", size="sm", variant="secondary")
                            with gr.Accordion("Video Analysis Results", open=False) as tb_analysis_accordion:
                                tb_video_analysis_output = gr.Textbox(
                                    container=False,
                                    lines=12,
                                    show_label=False,
                                    interactive=False
                                )


                    # --- Right Column: Output and Controls ---
                    with gr.Column(scale=1):
                        with gr.Tabs():
                            with gr.TabItem("Processed Video"):
                                processed_video = gr.Video(label="Toolbox Processed Video", interactive=False, elem_classes="video-window")
                        with gr.Row():
                            tb_use_as_input_btn = gr.Button("Use as Input", size="sm", scale=4)
                            initial_autosave_state = toolbox_processor.autosave_enabled
                            tb_manual_save_btn = gr.Button("Manual Save 💾", variant="secondary", size="sm", scale=4, visible=not initial_autosave_state)

                        # --- Settings & File Management Group ---
                        with gr.Group():
                            tb_open_folder_btn = gr.Button("📁 Open Outputs", scale=1, size="sm")
                            with gr.Row():
                                tb_autosave_checkbox = gr.Checkbox(label="Autosave", scale=1, value=initial_autosave_state)
                            with gr.Row():                               
                                tb_clear_temp_btn = gr.Button("🗑️ Clear Temp", size="sm", scale=1, variant="stop")
                            with gr.Row():
                                frames_output_quality = gr.Slider(
                                    minimum=1, maximum=10, step=1, value=8, label="Master Output Quality",
                                    info="(1=lowest, 10=highest/near-lossless). Quality 8 is recommended for most users. Quality 10 creates VERY large files. Note: The Export operation will re-encode with its own quality settings if selected in the pipeline."
                                )
                        with gr.Row():
                            tb_status_message = gr.Textbox(label="Toolbox Console", lines=8, interactive=False)                        

                        
                # --- Accordion for operation settings ---
                with gr.Accordion("Operations Settings", open=True):
                    with gr.Tabs():
                        # --- Frame Adjust Tab ---
                        with gr.TabItem("🎞️ Frame Adjust (Speed & Interpolation)"):
                            with gr.Row():
                                gr.Markdown("Adjust video speed and interpolate frames using RIFE AI.")
                            with gr.Row():
                                process_fps_mode = gr.Radio(
                                    choices=["No Interpolation", "2x Frames", "4x Frames"], value="2x Frames",  label="RIFE Frame Interpolation",
                                    info="Select '2x' or '4x' RIFE Interpolation to double or quadruple the frame rate, creating smoother motion. 4x is more intensive and runs the 2x process twice."
                                )
                                frames_use_streaming_checkbox = gr.Checkbox(
                                    label="Use Streaming (Low Memory Mode)", value=False,
                                    info="Enable for stable, low-memory RIFE on long videos. This avoids loading all frames into RAM. Note: 'Adjust Video Speed' is ignored in this mode."              
                                )
                            with gr.Row():
                                process_speed_factor = gr.Slider(
                                    minimum=0.5, maximum=2.0, step=0.05, value=1, label="Adjust Video Speed Factor",
                                    info="Values < 1.0 slow down the video, values > 1.0 speed it up. Affects video and audio."
                                )
                            process_frames_btn = gr.Button("🚀 Process Frames", variant="primary")

                        # --- Loop Tab ---
                        with gr.TabItem("🔄 Video Loop"):
                            with gr.Row():
                                gr.Markdown("Create looped or ping-pong versions of the video.")

                            loop_type_select = gr.Radio(choices=["loop", "ping-pong"], value="loop", label="Loop Type")
                            num_loops_slider = gr.Slider(
                                minimum=1, maximum=10, step=1, value=1, label="Number of Loops/Repeats",
                                info="The video will play its original content, then repeat this many additional times. E.g., 1 loop = 2 total plays of the segment."
                            )
                            create_loop_btn = gr.Button("🔁 Create Loop", variant="primary")
                            
                        # --- Export Tab ---
                        with gr.TabItem("📦 Compress, Encode & Export"):
                            with gr.Row():
                                with gr.Column(scale=2):
                                    export_format_radio = gr.Radio(
                                        ["MP4", "WebM", "GIF"], value="MP4", label="Output Format",
                                        info="MP4 is best for general use. WebM is great for web/Discord (smaller size). GIF is a widely-supported format for short, silent, looping clips. GIF output will always be saved."
                                    )
                                    export_quality_slider = gr.Slider(
                                        0, 100, value=85, step=1, label="Quality",
                                        info="Higher quality means a larger file size. 80-90 is a good balance for MP4/WebM."
                                    )
                                with gr.Column(scale=2):
                                    export_resize_slider = gr.Slider(
                                        256, 2048, value=1024, step=64, label="Max Width (pixels)",
                                        info="Resizes the video to this maximum width while maintaining aspect ratio. A powerful way to reduce file size."
                                    )
                                    export_name_input = gr.Textbox(
                                        label="Output Filename (optional)",
                                        value="",
                                        placeholder="e.g., my_final_video_for_discord",
                                                                        )
                            export_video_btn = gr.Button("🚀 Export Video", variant="primary")
                
        
            
        ### --- EVENT HANDLERS --- ###

        def do_sleep(delay_seconds=6):
            """
            Just sleeps. This will be used in the Gradio chain with no outputs 
            to prevent the UI from fading the target component.
            """
            time.sleep(delay_seconds)

        def get_random_idle_state():
            """Returns a random idle state HTML for the save_status display."""
            return random.choice(IDLE_STATES)

        def do_clear():
            """Returns a random idle state HTML instead of empty string."""
            return get_random_idle_state()
        
        def toggle_tiled_dit_options(is_checked):
            return gr.update(visible=is_checked)
        
        def update_clear_on_start_config(value):
            config = load_config()
            config["clear_temp_on_start"] = value
            save_config(config)
            status = "enabled" if value else "disabled"
            return f'<div style="padding: 1px; background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 1px; color: #0c5460;">ℹ️ Clear temp on start: {status}</div>'
        
        def update_autosave_config(value):
            config = load_config()
            config["autosave"] = value
            save_config(config)
            status = "enabled" if value else "disabled"
            return f'<div style="padding: 1px; background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 1px; color: #0c5460;">ℹ️ Autosave: {status}</div>'

        tiled_dit_checkbox.change(fn=toggle_tiled_dit_options, inputs=[tiled_dit_checkbox], outputs=[tiled_dit_options])
        
        autosave_checkbox.change(
            fn=update_autosave_config, 
            inputs=[autosave_checkbox], 
            outputs=[save_status]
        ).then(
            fn=do_sleep,
            inputs=None,
            outputs=None,
            show_progress="hidden"
        ).then(
            fn=do_clear,
            inputs=None,
            outputs=[save_status],
            show_progress="hidden"
        )
        
        clear_on_start_checkbox.change(
            fn=update_clear_on_start_config, 
            inputs=[clear_on_start_checkbox], 
            outputs=[save_status]
        ).then(
            fn=do_sleep,
            inputs=None,
            outputs=None,
            show_progress="hidden"
        ).then(
            fn=do_clear,
            inputs=None,
            outputs=[save_status],
            show_progress="hidden"
        )
        
        open_folder_button.click(
            fn=lambda: open_folder(OUTPUT_DIR), 
            inputs=[], 
            outputs=[save_status]
        ).then(
            fn=do_sleep,
            inputs=None,
            outputs=None,
            show_progress="hidden"
        ).then(
            fn=do_clear,
            inputs=None,
            outputs=[save_status],
            show_progress="hidden"
        )
        
        clear_temp_button.click(
            fn=clear_temp_files, 
            inputs=[], 
            outputs=[save_status]
        ).then(
            fn=do_sleep,
            inputs=None,
            outputs=None,
            show_progress="hidden"
        ).then(
            fn=do_clear,
            inputs=None,
            outputs=[save_status],
            show_progress="hidden"
        )
        
        save_button.click(
            fn=save_file_manually, 
            inputs=[output_file_path], 
            outputs=[save_status]
        ).then(
            fn=do_sleep,
            inputs=None,
            outputs=None,
            show_progress="hidden"
        ).then(
            fn=do_clear,
            inputs=None,
            outputs=[save_status],
            show_progress="hidden"
        )

        run_button.click(
            fn=run_flashvsr_integrated,
            inputs=[
                input_video, mode_radio, scale_slider, color_fix_checkbox, tiled_vae_checkbox,
                tiled_dit_checkbox, tile_size_slider, tile_overlap_slider, unload_dit_checkbox,
                dtype_radio, seed_number, device_textbox, fps_number, quality_slider, attention_mode_radio,
                sparse_ratio_slider, kv_ratio_slider, local_range_slider, autosave_checkbox
            ],
            outputs=[video_output, output_file_path, video_slider_output]
        )

        def update_monitor():
            return SystemMonitor.get_system_info()
            
        monitor_timer = gr.Timer(2, active=True)
        monitor_timer.tick(fn=update_monitor, outputs=[gpu_monitor, cpu_monitor]) 
        
        def send_to_toolbox(video_path):
            if not video_path:
                return gr.update(), gr.update(), '<div style="padding: 1px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 1px; color: #856404;">⚠️ No video to send!</div>'
            # Switches to tab 1 (Toolbox) and sets the input video value
            return gr.update(selected=1), gr.update(value=video_path), '<div style="padding: 1px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 1px; color: #155724;">✅ Video sent to Toolbox!</div>'

        send_to_toolbox_btn.click(
            fn=send_to_toolbox,
            inputs=[output_file_path],
            outputs=[main_tabs, tb_input_video, save_status]
        ).then(
            fn=do_sleep,
            inputs=None,
            outputs=None,
            show_progress="hidden"
        ).then(
            fn=do_clear,
            inputs=None,
            outputs=[save_status],
            show_progress="hidden"
        )        

        # --- Toolbox Tab Handlers ---
        
        tb_open_folder_btn.click(
            fn=toolbox_processor.open_output_folder, 
            outputs=[tb_status_message]
        )
        def handle_autosave_toggle(is_enabled):
            # Update toolbox processor
            message = toolbox_processor.set_autosave_mode(is_enabled)
            # Save to shared config
            config = load_config()
            config["tb_autosave"] = is_enabled
            save_config(config)
            return gr.update(visible=not is_enabled), message
        
        tb_autosave_checkbox.change(
            fn=handle_autosave_toggle,
            inputs=[tb_autosave_checkbox],
            outputs=[tb_manual_save_btn, tb_status_message]
        )
    
        def handle_single_operation(operation_func, video_path, status_message, **kwargs):
            if not video_path:
                return None, "⚠️ No input video found."

            temp_video = operation_func(video_path, progress=gr.Progress(), **kwargs)

            if not temp_video or temp_video == video_path:
                return video_path, f"❌ {status_message} failed. Check console."

            final_video_path = temp_video
            message = f"✅ {status_message} complete."

            if toolbox_processor.autosave_enabled:
                temp_path = Path(temp_video)
                final_path = toolbox_processor.output_dir / temp_path.name
                final_video_path = toolbox_processor._copy_to_permanent_storage(temp_video, final_path)
                message += f"\n✅ Autosaved result to: {final_path}"
            else:
                message += "\nℹ️ Autosave is off. Result is temporary. Use 'Manual Save'."
            
            return final_video_path, message        

        process_frames_btn.click(
            lambda video_path, status, fps, speed, stream, quality: handle_single_operation(toolbox_processor.adjust_frames, video_path, status, fps_mode=fps, speed_factor=speed, use_streaming=stream, output_quality=quality),
            inputs=[tb_input_video, gr.Textbox("Frame Adjustment", visible=False), process_fps_mode, process_speed_factor, frames_use_streaming_checkbox, frames_output_quality],
            outputs=[processed_video, tb_status_message]
        )
        
        def handle_create_loop(video_path, loop_type, num_loops, progress=gr.Progress()):
            if not video_path:
                return None, "⚠️ No video provided for loop creation."
            
            output_video = toolbox_processor.create_loop(video_path, loop_type, num_loops, progress)
            
            if output_video:
                message = f"✅ Loop created successfully: {os.path.basename(output_video)}"
                if toolbox_processor.autosave_enabled:
                    temp_path = Path(output_video)
                    final_path = toolbox_processor.output_dir / temp_path.name
                    final_video = toolbox_processor._copy_to_permanent_storage(output_video, final_path)
                    message += f"\n✅ Autosaved to: {final_path}"
                    return final_video, message
                else:
                    message += "\nℹ️ Autosave is off. Use 'Manual Save' to keep it."
                    return output_video, message
            else:
                return None, "❌ Loop creation failed. Check console for details."
    
    
        create_loop_btn.click(
            fn=handle_create_loop, 
            inputs=[tb_input_video, loop_type_select, num_loops_slider], 
            outputs=[processed_video, tb_status_message]
        )
        
        export_video_btn.click(
            lambda video_path, status, format, quality, width, name: handle_single_operation(toolbox_processor.export_video, video_path, status, export_format=format, quality=quality, max_width=width, output_name=name),
            inputs=[tb_input_video, gr.Textbox("Exporting", visible=False), export_format_radio, export_quality_slider, export_resize_slider, export_name_input],
            outputs=[processed_video, tb_status_message]
        )

        def handle_manual_save(video_path_from_player):
            if not video_path_from_player or not os.path.exists(video_path_from_player):
                 return "⚠️ No video in the output player to save."
            
            saved_path = toolbox_processor.save_video_from_any_source(video_path_from_player)
            
            if saved_path:
                return f"✅ Video successfully saved to: {saved_path}"
            else:
                return "❌ An error occurred during save. Check the console for details."

        tb_manual_save_btn.click(
            fn=handle_manual_save,
            inputs=[processed_video], # Takes input directly from the video player
            outputs=[tb_status_message]  # Only needs to update the status message
        )

        tb_open_folder_btn.click(
            fn=toolbox_processor.open_output_folder, 
            outputs=[tb_status_message]
        )

        # Track which input tab is active (Single vs Batch)
        # The select event passes a SelectData object with an 'index' attribute
        def update_tab_index(evt: gr.SelectData):
            return evt.index
        
        tb_input_tabs.select(
            fn=update_tab_index,
            inputs=[],
            outputs=[tb_active_tab_index]
        )

        # Analyze video button - also opens the accordion
        def analyze_and_open(video_path):
            analysis_result = toolbox_processor.analyze_video(video_path)
            return analysis_result, gr.update(open=True)
        
        tb_analyze_button.click(
            fn=analyze_and_open,
            inputs=[tb_input_video],
            outputs=[tb_video_analysis_output, tb_analysis_accordion]
        )

        # Wire up the pipeline button
        tb_start_pipeline_btn.click(
            fn=handle_start_pipeline,
            inputs=[
                tb_active_tab_index,
                tb_input_video,
                tb_batch_input_files,
                tb_pipeline_steps_chkbox,
                # Frame Adjust params
                process_fps_mode,
                process_speed_factor,
                frames_use_streaming_checkbox,
                frames_output_quality,
                # Video Loop params
                loop_type_select,
                num_loops_slider,
                # Export params
                export_format_radio,
                export_quality_slider,
                export_resize_slider,
                export_name_input
            ],
            outputs=[processed_video, tb_status_message]
        )

        # Use as Input button - sends processed video back to input
        def use_as_input(video_path):
            if not video_path:
                return None, "⚠️ No processed video to use as input."
            return video_path, "✅ Processed video loaded as input."
        
        tb_use_as_input_btn.click(
            fn=use_as_input,
            inputs=[processed_video],
            outputs=[tb_input_video, tb_status_message]
        )

        # Clear temp button
        def clear_toolbox_temp():
            try:
                if toolbox_processor.temp_dir.exists():
                    shutil.rmtree(toolbox_processor.temp_dir)
                    os.makedirs(toolbox_processor.temp_dir, exist_ok=True)
                    return "✅ Toolbox temp files cleared."
                return "ℹ️ Temp directory doesn't exist."
            except Exception as e:
                return f"❌ Error clearing temp files: {e}"
        
        tb_clear_temp_btn.click(
            fn=clear_toolbox_temp,
            outputs=[tb_status_message]
        )

    
    return demo

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check user preference for clearing temp on start
    config = load_config()
    if config.get("clear_temp_on_start", False):
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            log("Temp files cleared on startup.", message_type="info")
    
    os.makedirs(TEMP_DIR, exist_ok=True)
    model_download()
    ui = create_ui()
    if args.listen:
        ui.queue().launch(share=False, server_name="0.0.0.0", server_port=args.port)
    else:
        ui.queue().launch(share=False, server_port=args.port)