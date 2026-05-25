"""
Batch Video Frame Loader for ComfyUI
=====================================
A simple, reliable batch frame processor that:
1. Extracts all frames from a video to a temp folder on disk using FFmpeg
2. Loads only N frames at a time (batch_size), auto-advancing each run
3. Saves processed frames to disk with continuous numbering
4. Auto-requeues the workflow until all batches are processed

No VHS dependency. No generator magic. Just straightforward frame I/O
with a state file on disk to track progress.
"""

import os
import glob
import json
import hashlib
import math
import subprocess
import numpy as np
import torch
from PIL import Image
import urllib.request


class BatchVideoExtract:
    """
    Extracts ALL frames from a video file to a folder on disk.
    Run this once before processing. Frames are cached — if the same
    video was already extracted, it skips re-extraction.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "output_folder": ("STRING", {"default": "temp/batch_frames", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT",)
    RETURN_NAMES = ("frames_folder", "total_frames", "fps",)
    FUNCTION = "extract"
    CATEGORY = "Batch Video Processor"

    def extract(self, video_path, output_folder):
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Create a unique subfolder based on video path hash
        video_hash = hashlib.md5(video_path.encode()).hexdigest()[:12]
        frames_dir = os.path.join(output_folder, video_hash)
        os.makedirs(frames_dir, exist_ok=True)

        # Check if already extracted (marker file)
        marker = os.path.join(frames_dir, "_extraction_complete.json")
        if os.path.exists(marker):
            with open(marker, 'r') as f:
                info = json.load(f)
            print(f"[BatchVideoExtract] Using cached extraction: {info['total_frames']} frames at {info['fps']} fps")
            return (frames_dir, info['total_frames'], info['fps'],)

        # Get video FPS using ffprobe
        fps_cmd = [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "csv=p=0",
            video_path
        ]
        try:
            fps_result = subprocess.run(fps_cmd, capture_output=True, text=True, check=True)
            fps_str = fps_result.stdout.strip()
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
        except (subprocess.CalledProcessError, ValueError):
            fps = 24.0
            print(f"[BatchVideoExtract] Could not detect FPS, defaulting to {fps}")

        # Extract all frames as PNG
        print(f"[BatchVideoExtract] Extracting frames from: {video_path}")
        print(f"[BatchVideoExtract] Output folder: {frames_dir}")

        extract_cmd = [
            "ffmpeg", "-i", video_path,
            "-vsync", "0",
            "-frame_pts", "1",
            os.path.join(frames_dir, "frame_%06d.png")
        ]
        subprocess.run(extract_cmd, capture_output=True, check=True)

        # Count frames
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
        total_frames = len(frame_files)

        if total_frames == 0:
            raise RuntimeError(f"FFmpeg extracted 0 frames from {video_path}")

        # Write marker
        info = {
            "video_path": video_path,
            "total_frames": total_frames,
            "fps": fps,
            "video_hash": video_hash
        }
        with open(marker, 'w') as f:
            json.dump(info, f)

        print(f"[BatchVideoExtract] Extracted {total_frames} frames at {fps} fps")
        return (frames_dir, total_frames, fps,)


class BatchFrameLoader:
    """
    Loads a batch of frames from a folder on disk.
    
    AUTO-LOOP MODE (auto_loop=True, default):
    - Tracks progress in a state file on disk
    - First run processes batch 0, next run batch 1, etc.
    - Automatically re-queues the workflow after each batch
    - Stops when all batches are done
    - Set reset=True to start over from batch 0
    
    MANUAL MODE (auto_loop=False):
    - Uses manual_batch_index widget directly
    - No auto-requeue, you control which batch to process
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames_folder": ("STRING", {"default": "", "multiline": False}),
                "batch_size": ("INT", {"default": 24, "min": 1, "max": 9999, "step": 1}),
                "overlap": ("INT", {"default": 0, "min": 0, "max": 48, "step": 1}),
                "auto_loop": ("BOOLEAN", {"default": True}),
                "reset": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "manual_batch_index": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    @classmethod
    def IS_CHANGED(s, **kwargs):
        # Always re-execute - batch state changes on disk between runs
        import time
        return time.time()

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "BOOLEAN", "INT", "INT",)
    RETURN_NAMES = ("images", "batch_index", "total_batches", "frames_in_batch", "is_last_batch", "overlap", "nuke_frame_start",)
    FUNCTION = "load_batch"
    CATEGORY = "Batch Video Processor"

    def _state_file(self, frames_folder):
        return os.path.join(frames_folder, "_batch_state.json")

    def _read_state(self, frames_folder):
        sf = self._state_file(frames_folder)
        if os.path.exists(sf):
            with open(sf, 'r') as f:
                return json.load(f)
        return {"current_batch": 0}

    def _write_state(self, frames_folder, state):
        sf = self._state_file(frames_folder)
        with open(sf, 'w') as f:
            json.dump(state, f)

    def load_batch(self, frames_folder, batch_size, overlap, auto_loop, reset,
                   manual_batch_index=0, unique_id=None):
        if not os.path.isdir(frames_folder):
            raise FileNotFoundError(f"Frames folder not found: {frames_folder}")

        # Get sorted frame files
        frame_files = sorted(glob.glob(os.path.join(frames_folder, "frame_*.png")))
        total_frames = len(frame_files)

        if total_frames == 0:
            raise RuntimeError(f"No frame_*.png files found in {frames_folder}")

        step_size = max(1, batch_size - overlap)
        if total_frames <= batch_size:
            total_batches = 1
        else:
            total_batches = math.ceil((total_frames - overlap) / step_size)

        # Determine which batch to process
        if auto_loop:
            state = self._read_state(frames_folder)
            if reset:
                state["current_batch"] = 0
                self._write_state(frames_folder, state)
                print(f"[BatchFrameLoader] Reset to batch 0")

            batch_index = state["current_batch"]
        else:
            batch_index = manual_batch_index

        # Clamp batch_index
        if batch_index >= total_batches:
            print(f"[BatchFrameLoader] All {total_batches} batches already processed!")
            batch_index = total_batches - 1

        is_last_batch = (batch_index >= total_batches - 1)

        # Calculate frame range with sliding window
        start_idx = batch_index * step_size
        end_idx = min(start_idx + batch_size, total_frames)
        batch_files = frame_files[start_idx:end_idx]
        frames_in_batch = len(batch_files)

        print(f"[BatchFrameLoader] ========================================")
        print(f"[BatchFrameLoader] BATCH {batch_index + 1} of {total_batches}")
        print(f"[BatchFrameLoader] Frames {start_idx} to {end_idx - 1} ({frames_in_batch} frames, overlap={overlap})")
        print(f"[BatchFrameLoader] ========================================")

        # Load frames into tensor
        images = []
        for fpath in batch_files:
            img = Image.open(fpath).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            images.append(img_np)

        images_tensor = torch.from_numpy(np.stack(images, axis=0))

        # Auto-loop: advance state and re-queue
        if auto_loop:
            state["current_batch"] = batch_index + 1
            self._write_state(frames_folder, state)

            if not is_last_batch:
                print(f"[BatchFrameLoader] >> Auto-requeuing for batch {batch_index + 2}/{total_batches}...")
                import threading
                def delayed_requeue():
                    import time
                    # Wait until the queue is empty (current execution finished)
                    for attempt in range(600):
                        time.sleep(1)
                        try:
                            resp = urllib.request.urlopen('http://127.0.0.1:8188/queue')
                            queue = json.loads(resp.read())
                            if not queue.get('queue_running') and not queue.get('queue_pending'):
                                # Queue is empty, execution finished. Grab the last prompt from history
                                time.sleep(1)  # extra 1s buffer
                                resp2 = urllib.request.urlopen('http://127.0.0.1:8188/history?max_items=1')
                                history = json.loads(resp2.read())
                                if history:
                                    last_id = list(history.keys())[0]
                                    entry = history[last_id]
                                    prompt_nodes = entry['prompt'][2]
                                    payload = json.dumps({"prompt": prompt_nodes}).encode('utf-8')
                                    req = urllib.request.Request(
                                        'http://127.0.0.1:8188/prompt',
                                        data=payload,
                                        headers={'Content-Type': 'application/json'},
                                        method='POST'
                                    )
                                    urllib.request.urlopen(req)
                                    print(f"[BatchFrameLoader] >> Requeued OK after {attempt+1}s wait")
                                    return
                        except Exception:
                            pass
                    print(f"[BatchFrameLoader] !! Requeue timed out after 600s")
                t = threading.Thread(target=delayed_requeue, daemon=True)
                t.start()
            else:
                print(f"[BatchFrameLoader] ✓ ALL BATCHES COMPLETE ({total_batches} batches, {total_frames} frames)")
                # Reset state for next time
                state["current_batch"] = 0
                self._write_state(frames_folder, state)

        nuke_frame_start = start_idx + 1  # 1-indexed for NukeWrite
        return (images_tensor, batch_index, total_batches, frames_in_batch, is_last_batch, overlap, nuke_frame_start,)


class BatchFrameSaver:
    """
    Saves a batch of processed IMAGE frames to disk as individual files.
    
    Automatically numbers them based on batch_index and batch_size
    so all batches produce a continuous sequence.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_folder": ("STRING", {"default": "output/cinema_raw", "multiline": False}),
                "batch_size": ("INT", {"default": 24, "min": 1, "max": 9999, "step": 1}),
                "overlap": ("INT", {"default": 0, "min": 0, "max": 48, "step": 1}),
                "format": (["exr_16bit", "png_16bit", "png_8bit", "tiff_16bit"],),
                "filename_prefix": ("STRING", {"default": "frame"}),
            },
            "optional": {
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("output_folder", "frames_saved",)
    FUNCTION = "save_batch"
    CATEGORY = "Batch Video Processor"
    OUTPUT_NODE = True

    def save_batch(self, images, output_folder, batch_size, overlap, format, filename_prefix, batch_index=0):
        os.makedirs(output_folder, exist_ok=True)

        counter_file = os.path.join(output_folder, "_frame_counter.json")

        if batch_index == 0:
            start_frame = 0
            with open(counter_file, 'w') as f:
                json.dump({"count": 0}, f)
        else:
            if os.path.exists(counter_file):
                with open(counter_file, 'r') as f:
                    start_frame = json.load(f)["count"]
            else:
                start_frame = 0

        images_to_save = images[overlap:] if batch_index > 0 else images
        batch_count = images_to_save.shape[0]

        print(f"[BatchFrameSaver] Saving {batch_count} frames starting at {start_frame} "
              f"(skipping {overlap if batch_index > 0 else 0} overlap frames) to {output_folder} as {format}")

        for i in range(batch_count):
            frame_num = start_frame + i
            img_tensor = images_to_save[i]

            if format == "png_16bit":
                img_np_8 = (img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                filepath = os.path.join(output_folder, f"{filename_prefix}_{frame_num:06d}.png")
                img = Image.fromarray(img_np_8, mode='RGB')
                img.save(filepath)

            elif format == "png_8bit":
                img_np = (img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                filepath = os.path.join(output_folder, f"{filename_prefix}_{frame_num:06d}.png")
                img = Image.fromarray(img_np, mode='RGB')
                img.save(filepath)

            elif format == "exr_16bit":
                img_np = img_tensor.cpu().numpy().astype(np.float32)
                filepath = os.path.join(output_folder, f"{filename_prefix}_{frame_num:06d}.exr")
                try:
                    import OpenImageIO as oiio
                    h, w, c = img_np.shape
                    spec = oiio.ImageSpec(w, h, c, oiio.HALF)
                    out = oiio.ImageOutput.create(filepath)
                    out.open(filepath, spec)
                    out.write_image(img_np)
                    out.close()
                except ImportError:
                    _write_minimal_exr(filepath, img_np)

            elif format == "tiff_16bit":
                img_np = (img_tensor.cpu().numpy() * 65535.0).clip(0, 65535).astype(np.uint16)
                filepath = os.path.join(output_folder, f"{filename_prefix}_{frame_num:06d}.tiff")
                img = Image.fromarray(img_np, mode='I;16')
                img.save(filepath)

        with open(counter_file, 'w') as f:
            json.dump({"count": start_frame + batch_count}, f)

        print(f"[BatchFrameSaver] ✓ Saved {batch_count} frames ({start_frame} to {start_frame + batch_count - 1})")
        return (output_folder, batch_count,)


def _write_minimal_exr(filepath, img_np):
    """Minimal EXR writer for float32 RGB data without OpenImageIO dependency."""
    import struct

    h, w, c = img_np.shape
    assert c == 3, "Expected RGB image"

    half_data = img_np.astype(np.float16)

    magic = struct.pack('<I', 20000630)
    version = struct.pack('<I', 2)

    header = b''

    def add_attr(name, type_name, value):
        nonlocal header
        header += name.encode() + b'\x00'
        header += type_name.encode() + b'\x00'
        header += struct.pack('<I', len(value))
        header += value

    channels_data = b''
    for ch_name in ['B', 'G', 'R']:
        channels_data += ch_name.encode() + b'\x00'
        channels_data += struct.pack('<I', 1)
        channels_data += struct.pack('<I', 0)
        channels_data += struct.pack('<i', 0)
        channels_data += struct.pack('<i', 1)
        channels_data += struct.pack('<i', 1)
    channels_data += b'\x00'
    add_attr('channels', 'chlist', channels_data)

    add_attr('compression', 'compression', struct.pack('<B', 0))
    add_attr('dataWindow', 'box2i', struct.pack('<iiii', 0, 0, w - 1, h - 1))
    add_attr('displayWindow', 'box2i', struct.pack('<iiii', 0, 0, w - 1, h - 1))
    add_attr('lineOrder', 'lineOrder', struct.pack('<B', 0))
    add_attr('pixelAspectRatio', 'float', struct.pack('<f', 1.0))
    add_attr('screenWindowCenter', 'v2f', struct.pack('<ff', 0.0, 0.0))
    add_attr('screenWindowWidth', 'float', struct.pack('<f', 1.0))

    header += b'\x00'

    header_size = len(magic) + len(version) + len(header)
    offset_table_size = h * 8
    data_start = header_size + offset_table_size

    offsets = []
    scanline_data = b''
    current_offset = data_start

    for y in range(h):
        row = half_data[y]
        b_data = row[:, 2].tobytes()
        g_data = row[:, 1].tobytes()
        r_data = row[:, 0].tobytes()
        pixel_data = b_data + g_data + r_data

        block = struct.pack('<i', y) + struct.pack('<I', len(pixel_data)) + pixel_data
        offsets.append(current_offset)
        scanline_data += block
        current_offset += len(block)

    offset_table = b''.join(struct.pack('<Q', o) for o in offsets)

    with open(filepath, 'wb') as f:
        f.write(magic)
        f.write(version)
        f.write(header)
        f.write(offset_table)
        f.write(scanline_data)


NODE_CLASS_MAPPINGS = {
    "BatchVideoExtract": BatchVideoExtract,
    "BatchFrameLoader": BatchFrameLoader,
    "BatchFrameSaver": BatchFrameSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchVideoExtract": "Batch Video Extract 🎬",
    "BatchFrameLoader": "Batch Frame Loader 🎬",
    "BatchFrameSaver": "Batch Frame Saver 🎬",
}
