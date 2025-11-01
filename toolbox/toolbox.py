import os
import gc
import sys
import subprocess
import types
import torch
import re
import numpy as np
import imageio
import gradio as gr
import shutil
import traceback
import math
import json
from datetime import datetime
from pathlib import Path

import devicetorch

# Local imports for RIFE and ESRGAN
from toolbox.rife_core import RIFEHandler

device_name_str = devicetorch.get(torch)

class ToolboxProcessor:
    """
    A processor for handling upscale, frame adjustment, and export operations.
    """
    def __init__(self, autosave_enabled=True):
        self.device_obj = torch.device(device_name_str)
        # toolbox.py is in /app/toolbox, so parent.parent gets us to /app
        app_dir = Path(__file__).parent.parent.absolute()
        
        self.output_dir = app_dir / "outputs" / "toolbox"
        self.temp_dir = app_dir / "_temp" / "toolbox"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        self.ffmpeg_exe, self.ffprobe_exe, self.has_ffmpeg = self._initialize_ffmpeg()
        self.autosave_enabled = autosave_enabled
        self.rife_handler = RIFEHandler()
        
    def set_autosave_mode(self, is_enabled):
        """Updates the autosave mode."""
        self.autosave_enabled = is_enabled
        status = "ON" if is_enabled else "OFF"
        return f"✅ Autosave is now {status}."

    def save_video_from_any_source(self, video_source_path):
        """
        Copies a video from the toolbox temp directory to the permanent output folder,
        preserving its filename. This is the backend for the Manual Save button.
        """
        try:
            # Get the filename from the source path
            source_filename = Path(video_source_path).name
            
            # Create the destination path
            destination_path = self.output_dir / source_filename
            
            print(f"Copying video from '{video_source_path}' to '{destination_path}'")
            
            # Copy the file to preserve it in temp for further operations
            shutil.copy2(video_source_path, destination_path)
            
            return str(destination_path)
            
        except Exception as e:
            print(f"Error during manual save: {e}\n{traceback.format_exc()}")
            return None
            
    def open_output_folder(self):
        """Opens the toolbox output folder in the system's file explorer."""
        folder_path = os.path.abspath(self.output_dir)
        try:
            if sys.platform == "win32": os.startfile(folder_path)
            elif sys.platform == "darwin": subprocess.Popen(["open", folder_path])
            else: subprocess.Popen(["xdg-open", folder_path])
            return f"Opened output folder: {folder_path}"
        except Exception as e: return f"❌ Error opening folder: {e}"

    def analyze_video(self, video_path):
        """Analyzes video file and returns detailed information."""
        if not video_path:
            return "⚠️ No video provided for analysis."
        
        resolved_path = str(Path(video_path).resolve())
        report = []
        
        # Get file size
        file_size_display = "N/A"
        try:
            if os.path.exists(resolved_path):
                size_bytes = os.path.getsize(resolved_path)
                if size_bytes < 1024:
                    file_size_display = f"{size_bytes} B"
                elif size_bytes < 1024**2:
                    file_size_display = f"{size_bytes/1024:.2f} KB"
                elif size_bytes < 1024**3:
                    file_size_display = f"{size_bytes/1024**2:.2f} MB"
                else:
                    file_size_display = f"{size_bytes/1024**3:.2f} GB"
        except Exception as e:
            print(f"Warning: Could not get file size: {e}")
        
        # Initialize variables
        video_width, video_height = 0, 0
        num_frames_value = None
        duration_display, fps_display, resolution_display = "N/A", "N/A", "N/A"
        nframes_display, has_audio_str = "N/A", "No"
        analysis_source = "imageio"
        
        # Try ffprobe first if available
        if self.has_ffmpeg and self.ffprobe_exe:
            try:
                probe_cmd = [
                    self.ffprobe_exe, "-v", "error", "-show_format", "-show_streams",
                    "-of", "json", resolved_path
                ]
                result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True, errors='ignore')
                probe_data = json.loads(result.stdout)
                
                video_stream = next((s for s in probe_data.get("streams", []) if s.get("codec_type") == "video"), None)
                audio_stream = next((s for s in probe_data.get("streams", []) if s.get("codec_type") == "audio"), None)
                
                if video_stream:
                    analysis_source = "ffprobe"
                    
                    # Duration
                    duration_str = probe_data.get("format", {}).get("duration", "0")
                    duration = float(duration_str) if duration_str and duration_str.replace('.', '', 1).isdigit() else 0.0
                    duration_display = f"{duration:.2f} seconds"
                    
                    # FPS
                    def parse_fps(fps_s):
                        if isinstance(fps_s, (int, float)):
                            return float(fps_s)
                        if isinstance(fps_s, str) and "/" in fps_s:
                            try:
                                num, den = map(float, fps_s.split('/'))
                                return num / den if den != 0 else 0.0
                            except ValueError:
                                return 0.0
                        try:
                            return float(fps_s)
                        except ValueError:
                            return 0.0
                    
                    r_frame_rate = video_stream.get("r_frame_rate", "0/0")
                    avg_frame_rate = video_stream.get("avg_frame_rate", "0/0")
                    r_fps = parse_fps(r_frame_rate)
                    avg_fps = parse_fps(avg_frame_rate)
                    
                    calculated_fps = 0.0
                    if r_fps > 0:
                        calculated_fps = r_fps
                        fps_display = f"{r_fps:.2f} FPS"
                    if avg_fps > 0 and abs(r_fps - avg_fps) > 0.01:
                        calculated_fps = avg_fps
                        fps_display = f"{avg_fps:.2f} FPS (Avg, r: {r_fps:.2f})"
                    elif avg_fps > 0 and r_fps <= 0:
                        calculated_fps = avg_fps
                        fps_display = f"{avg_fps:.2f} FPS (Average)"
                    
                    # Resolution
                    video_width = video_stream.get("width", 0)
                    video_height = video_stream.get("height", 0)
                    resolution_display = f"{video_width}x{video_height}" if video_width and video_height else "N/A"
                    
                    # Frame count
                    nframes_str = video_stream.get("nb_frames")
                    if nframes_str and nframes_str.isdigit():
                        num_frames_value = int(nframes_str)
                        nframes_display = str(num_frames_value)
                    elif duration > 0 and calculated_fps > 0:
                        num_frames_value = int(duration * calculated_fps)
                        nframes_display = f"{num_frames_value} (Calculated)"
                    
                    # Audio
                    if audio_stream:
                        has_audio_str = (
                            f"Yes (Codec: {audio_stream.get('codec_name', 'N/A')}, "
                            f"Channels: {audio_stream.get('channels', 'N/A')}, "
                            f"Rate: {audio_stream.get('sample_rate', 'N/A')} Hz)"
                        )
                    
                    print("Video analysis complete (using ffprobe).")
            except Exception as e:
                print(f"ffprobe analysis failed, falling back to imageio: {e}")
                analysis_source = "imageio"
        
        # Fallback to imageio
        if analysis_source == "imageio":
            reader = None
            try:
                reader = imageio.get_reader(resolved_path)
                meta = reader.get_meta_data()
                
                # Duration
                duration_val = meta.get('duration')
                duration_display = f"{float(duration_val):.2f} seconds" if duration_val is not None else "N/A"
                
                # FPS
                fps_val = meta.get('fps')
                fps_display = f"{float(fps_val):.2f} FPS" if fps_val is not None else "N/A"
                
                # Resolution
                size_val = meta.get('size')
                if isinstance(size_val, tuple) and len(size_val) == 2:
                    video_width, video_height = int(size_val[0]), int(size_val[1])
                    resolution_display = f"{video_width}x{video_height}"
                
                # Frame count
                nframes_val = meta.get('nframes')
                if nframes_val not in [float('inf'), "N/A", None] and isinstance(nframes_val, (int, float)):
                    num_frames_value = int(nframes_val)
                    nframes_display = str(num_frames_value)
                elif hasattr(reader, 'count_frames'):
                    try:
                        nframes_count = reader.count_frames()
                        if nframes_count != float('inf'):
                            num_frames_value = int(nframes_count)
                            nframes_display = f"{num_frames_value} (Counted)"
                        else:
                            nframes_display = "Unknown (Stream)"
                    except Exception:
                        nframes_display = "Unknown"
                
                has_audio_str = "(Audio info not available via imageio)"
                print("Video analysis complete (using imageio).")
            except Exception as e:
                print(f"Error analyzing video with imageio: {e}")
                return f"❌ Error analyzing video: {e}"
            finally:
                if reader:
                    reader.close()
        
        # Build report
        report.append(f"📊 Video Analysis ({analysis_source})")
        report.append(f"File: {os.path.basename(video_path)}")
        report.append("─" * 50)
        report.append(f"📦 File Size: {file_size_display}")
        report.append(f"⏱️  Duration: {duration_display}")
        report.append(f"🎬 Frame Rate: {fps_display}")
        report.append(f"📐 Resolution: {resolution_display}")
        report.append(f"🎞️  Frames: {nframes_display}")
        report.append(f"🔊 Audio: {has_audio_str}")
        
        # Add upscale advisory
        # if video_width > 0 and video_height > 0:
            # HD_WIDTH = 1920
            # FOUR_K_WIDTH = 3800
            
            # is_hd_or_larger = video_width >= HD_WIDTH or video_height >= (HD_WIDTH * 9/16 * 0.95)
            # is_4k_or_larger = video_width >= FOUR_K_WIDTH or video_height >= (FOUR_K_WIDTH * 9/16 * 0.95)
            
            # warnings = []
            # if is_4k_or_larger:
                # warnings.append("This video is 4K+ resolution. Upscaling will be very slow and memory-intensive.")
            # elif is_hd_or_larger:
                # warnings.append("This video is HD or larger. Upscaling will be resource-intensive.")
            
            # if num_frames_value and num_frames_value > 900:
                # warnings.append(f"With {num_frames_value} frames, processing will be time-consuming.")
            
            # if warnings:
                # report.append("\n⚠️  PROCESSING ADVISORY")
                # report.append("─" * 50)
                # for warning in warnings:
                    # report.append(f"• {warning}")
        
        return "\n".join(report)

        
    def _initialize_ffmpeg(self):
        """Finds FFmpeg/FFprobe and sets status flags."""
        ffmpeg_path, ffprobe_path = self._find_ffmpeg_executables()
        has_ffmpeg = bool(ffmpeg_path) and bool(ffprobe_path)
        if not has_ffmpeg: print("WARNING: FFmpeg or FFprobe not found. Audio handling and some export formats will be disabled.")
        return ffmpeg_path, ffprobe_path, has_ffmpeg

    def _find_ffmpeg_executables(self):
        """Finds ffmpeg and ffprobe, prioritizing system PATH then imageio."""
        ffmpeg_path = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
        ffprobe_path = shutil.which("ffprobe") or shutil.which("ffprobe.exe")
        if not ffmpeg_path:
            try:
                imageio_ffmpeg_exe = imageio.plugins.ffmpeg.get_exe()
                if os.path.isfile(imageio_ffmpeg_exe): ffmpeg_path = imageio_ffmpeg_exe
            except Exception: pass
        return ffmpeg_path, ffprobe_path

    def _clean_filename(self, filename):
        """
        Intelligently handles timestamps in filenames.
        - Preserves original timestamps from unprocessed files
        - For processed files, removes only the timestamp added by the most recent operation
        """
        # Check if this filename already contains operation suffixes
        operation_patterns = ['upscaled_', 'frames_', 'exported_']
        has_operations = any(pattern in filename for pattern in operation_patterns)
        
        if has_operations:
            # Find all timestamps in the filename
            timestamp_pattern = r'_\d{8}_\d{6}'
            timestamps = re.findall(timestamp_pattern, filename)
            
            if len(timestamps) > 1:
                # Multiple timestamps - remove only the last one (most recent operation)
                # Replace the last timestamp with empty string
                last_timestamp = timestamps[-1]
                filename = filename.replace(last_timestamp, '')
            elif len(timestamps) == 1:
                # Only one timestamp - this might be original, but since we have operations,
                # it's likely from a previous operation, so remove it
                filename = re.sub(timestamp_pattern, '', filename)
        # If no operations yet, preserve all timestamps (original file)
        
        return filename.strip('_')

    def _generate_output_path(self, input_path, suffix, ext=".mp4", is_temp=False, batch_folder=None):
        """Generates a unique output path for processed videos, adapted from reference code."""
        base_name = self._clean_filename(Path(input_path).stem)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_name}_{suffix}_{timestamp}{ext}"
        if is_temp:
            os.makedirs(self.temp_dir, exist_ok=True)
            return self.temp_dir / filename
        if batch_folder:
            target_dir = self.output_dir / batch_folder
            os.makedirs(target_dir, exist_ok=True)
            return target_dir / filename
        os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir / filename

    def _copy_to_permanent_storage(self, temp_path, final_path):
        """Copies a temp file to permanent storage and cleans up the source temp file."""
        try:
            shutil.copy(temp_path, final_path)
            os.remove(temp_path)
            return str(final_path)
        except Exception as e:
            print(f"Error moving file to permanent storage: {e}")
            return str(temp_path)

    def _get_video_frame_count(self, video_path):
        """Uses ffprobe to get an accurate frame count."""
        if not self.has_ffmpeg: return None
        try:
            cmd = [self.ffprobe_exe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                   "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", video_path]
            return int(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip())
        except Exception: return None

    def _has_audio_stream(self, video_path):
        """Checks if a video file has an audio stream using ffprobe."""
        if not self.has_ffmpeg: return False
        try:
            cmd = [self.ffprobe_exe, "-v", "error", "-select_streams", "a:0",
                   "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path]
            return "audio" in subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip().lower()
        except Exception: return False

    def adjust_frames(self, video_path, fps_mode, speed_factor, use_streaming, output_quality=10, progress=gr.Progress()):
        if not video_path: print("No input video for frame adjustment."); return None
        
        interpolation_factor = 1
        if "2x" in fps_mode: interpolation_factor = 2
        elif "4x" in fps_mode: interpolation_factor = 4
        should_interpolate = interpolation_factor > 1

        if not should_interpolate and speed_factor == 1.0:
            print("INFO: No frame interpolation or speed change requested. Skipping frame adjustment.")
            return video_path

        temp_video_path = None
        try:
            print(f"Adjusting frames: Mode={fps_mode}, Speed={speed_factor}x, Streaming: {use_streaming}, Quality: {output_quality}/10")
            reader = imageio.get_reader(video_path)
            fps = reader.get_meta_data()['fps']
            output_fps = fps * interpolation_factor
            if use_streaming and speed_factor != 1.0:
                print("Note: Speed adjustment is ignored in RIFE streaming mode.")
                speed_factor = 1.0
            
            if use_streaming and should_interpolate:
                self.rife_handler._ensure_model_downloaded_and_loaded()
                temp_video_path = self._generate_output_path(video_path, "frames_temp", is_temp=True)
                writer = imageio.get_writer(temp_video_path, fps=output_fps, quality=output_quality)
                frame_iterator = iter(reader)
                frame1 = next(frame_iterator, None)
                if frame1 is not None:
                    desc = f"Interpolating Frames ({interpolation_factor}x Streaming)"
                    for frame2 in progress.tqdm(frame_iterator, desc=desc):
                        writer.append_data(frame1)
                        middle = self.rife_handler.interpolate_between_frames(frame1, frame2)
                        if middle is not None: writer.append_data(middle)
                        frame1 = frame2
                    writer.append_data(frame1)
                writer.close()
            else:
                frames = [frame for frame in reader]
                processed_frames = frames
                if speed_factor != 1.0:
                    print(f"Adjusting speed by {speed_factor}x (in-memory)...")
                    new_len = int(len(frames) / speed_factor)
                    indices = np.linspace(0, len(frames) - 1, new_len).astype(int)
                    processed_frames = [frames[i] for i in indices]
                if should_interpolate and len(processed_frames) > 1:
                    self.rife_handler._ensure_model_downloaded_and_loaded()
                    num_passes = int(math.log2(interpolation_factor))
                    for p in range(num_passes):
                        print(f"INFO: Starting RIFE interpolation pass {p + 1}/{num_passes}...")
                        interpolated_this_pass = []
                        desc = f"RIFE Pass {p+1}/{num_passes}"
                        frame_iterator = progress.tqdm(range(len(processed_frames) - 1), desc=desc)
                        for i in frame_iterator:
                            interpolated_this_pass.append(processed_frames[i])
                            middle = self.rife_handler.interpolate_between_frames(processed_frames[i], processed_frames[i+1])
                            interpolated_this_pass.append(middle if middle is not None else processed_frames[i])
                        interpolated_this_pass.append(processed_frames[-1])
                        processed_frames = interpolated_this_pass
                temp_video_path = self._generate_output_path(video_path, "frames_temp", is_temp=True)
                imageio.mimwrite(temp_video_path, processed_frames, fps=output_fps, quality=output_quality)
            reader.close()

            # --- Suffix and Final Path Generation ---
            suffix_parts = []
            if should_interpolate: suffix_parts.append(fps_mode.replace(' ', ''))
            if speed_factor != 1.0: suffix_parts.append(f"{speed_factor}x")
            suffix = f"frames_{'_'.join(suffix_parts)}"
            final_temp_output = self._generate_output_path(video_path, suffix, is_temp=True)

            # --- CORRECTED AUDIO MUXING LOGIC ---
            if self.has_ffmpeg and self._has_audio_stream(video_path):
                print("Muxing audio into processed video...")
                mux_cmd = [
                    self.ffmpeg_exe, "-y",
                    "-i", str(temp_video_path),
                    "-i", video_path,
                    "-c:v", "copy"
                ]
                
                # Conditionally apply the atempo filter ONLY if speed is changed
                if speed_factor != 1.0:
                    print(f"Applying atempo speed filter: {speed_factor}x")
                    audio_filters = [f"atempo={speed_factor}"]
                    if speed_factor > 2.0: audio_filters = [f"atempo=2.0,atempo={speed_factor/2.0}"]
                    if speed_factor < 0.5: audio_filters = [f"atempo=0.5,atempo={speed_factor/0.5}"]
                    mux_cmd.extend(["-filter:a", ",".join(audio_filters)])
                    mux_cmd.extend(["-c:a", "aac", "-b:a", "192k"]) # Re-encode when filtering
                else:
                    # If just interpolating, copy the audio directly
                    print("Copying original audio without speed change.")
                    mux_cmd.extend(["-c:a", "copy"])

                mux_cmd.extend([
                    "-map", "0:v:0", "-map", "1:a:0?",
                    "-shortest", str(final_temp_output)
                ])
                
                subprocess.run(mux_cmd, check=True, capture_output=True, text=True)
                if os.path.exists(temp_video_path): os.remove(temp_video_path)
            else:
                # This block runs if there's no FFmpeg or no original audio
                shutil.move(temp_video_path, final_temp_output)

            return str(final_temp_output)
        except Exception as e:
            print(f"Error during frame adjustment: {e}\n{traceback.format_exc()}")
            return video_path
        finally:
            self.rife_handler.unload_model()
            if temp_video_path and os.path.exists(temp_video_path): os.remove(temp_video_path)
            gc.collect(); torch.cuda.empty_cache()

    def create_loop(self, video_path, loop_type, num_loops, progress=gr.Progress()):
        """Creates a looped or ping-pong version of the video."""
        if video_path is None:
            print("No input video for loop creation.")
            return None
        if not self.has_ffmpeg:
            print("FFmpeg is required for creating video loops. This operation cannot proceed.")
            return video_path
        if loop_type == "none":
            print("Loop type 'none'. No action.")
            return video_path

        progress(0, desc="Initializing loop creation...")
        resolved_video_path = str(Path(video_path).resolve())
        output_path = self._generate_output_path(
            resolved_video_path, 
            suffix=f"loop_{loop_type}_{num_loops}x",
            is_temp=True
        )
        
        print(f"Creating {loop_type} ({num_loops}x) for {os.path.basename(resolved_video_path)}...")
        
        ping_pong_unit_path = None 
        original_video_has_audio = self._has_audio_stream(resolved_video_path)

        try:
            progress(0.2, desc=f"Preparing {loop_type} loop...")
            if loop_type == "ping-pong":
                ping_pong_unit_path = self._generate_output_path(
                    resolved_video_path, 
                    suffix="pingpong_unit_temp",
                    is_temp=True
                )
                # Create video-only ping-pong unit first
                ffmpeg_pp_unit_cmd = [
                    self.ffmpeg_exe, "-y", "-loglevel", "error",
                    "-i", resolved_video_path,
                    "-vf", "split[main][tmp];[tmp]reverse[rev];[main][rev]concat=n=2:v=1:a=0",
                    "-an", str(ping_pong_unit_path)
                ]
                subprocess.run(ffmpeg_pp_unit_cmd, check=True, capture_output=True, text=True)
                print(f"Created ping-pong unit (video-only): {ping_pong_unit_path}")

                ffmpeg_cmd = [
                    self.ffmpeg_exe, "-y", "-loglevel", "error",
                    "-stream_loop", str(num_loops - 1),
                    "-i", str(ping_pong_unit_path)
                ]

                if original_video_has_audio:
                    print("Original video has audio. Will loop audio for ping-pong.")
                    audio_loop_count_for_ffmpeg = (num_loops * 2) - 1
                    ffmpeg_cmd.extend([
                        "-i", resolved_video_path,
                        "-filter_complex", f"[1:a]areverse[areva];[1:a][areva]concat=n=2:v=0:a=1[ppa];[ppa]aloop=loop={num_loops-1}:size=2147483647[a_looped]",
                        "-map", "0:v:0", "-map", "[a_looped]",
                        "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-shortest"
                    ])
                else:
                    print("No audio in original or detection issue. Creating video-only ping-pong loop.")
                    ffmpeg_cmd.extend(["-c:v", "copy", "-an"])

                ffmpeg_cmd.append(str(output_path))

            else:  # Regular 'loop'
                ffmpeg_stream_loop_value = num_loops 
                
                if ffmpeg_stream_loop_value < 0: 
                    ffmpeg_stream_loop_value = 0

                total_plays = ffmpeg_stream_loop_value + 1
                print(f"Regular loop: original video + {ffmpeg_stream_loop_value} additional repeat(s). Total {total_plays} plays.")
                
                ffmpeg_cmd = [
                    self.ffmpeg_exe, "-y", "-loglevel", "error",
                    "-stream_loop", str(ffmpeg_stream_loop_value),
                    "-i", resolved_video_path,
                    "-c:v", "copy" 
                ]
                if original_video_has_audio:
                    print("Original video has audio. Re-encoding to AAC for looped MP4 (if not already AAC).")
                    ffmpeg_cmd.extend(["-c:a", "aac", "-b:a", "192k", "-map", "0:v:0", "-map", "0:a:0?"])
                else:
                    print("No audio in original or detection issue. Looped video will be silent.")
                    ffmpeg_cmd.extend(["-an", "-map", "0:v:0"])
                ffmpeg_cmd.append(str(output_path))
            
            print(f"Processing video {loop_type} with FFmpeg...")
            progress(0.5, desc=f"Running FFmpeg for {loop_type}...")
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, errors='ignore')

            progress(1.0, desc=f"{loop_type.capitalize()} loop created successfully.")
            print(f"Loop creation complete: {output_path}")
            return str(output_path)
        except subprocess.CalledProcessError as e_loop:
            print(f"FFmpeg error during {loop_type} creation: {e_loop}")
            if e_loop.stderr:
                print(f"FFmpeg stderr: {e_loop.stderr}")
            progress(1.0, desc=f"Error creating {loop_type}.")
            return None
        except Exception as e:
            print(f"Error creating loop: {e}")
            print(traceback.format_exc())
            progress(1.0, desc="Error creating loop.")
            return None
        finally:
            if ping_pong_unit_path and os.path.exists(ping_pong_unit_path):
                try:
                    os.remove(ping_pong_unit_path)
                except Exception as e_clean_pp:
                    print(f"Could not remove temp ping-pong unit: {e_clean_pp}")
            gc.collect()
            
    def export_video(self, video_path, export_format, quality, max_width, output_name, progress=gr.Progress()):
        if not video_path: print("No input video to export."); return None
        if not self.has_ffmpeg: print("FFmpeg is required for export."); return None
        print(f"Exporting video to {export_format} with quality {quality} and max width {max_width}px.")
        try:
            ext = f".{export_format.lower()}"
            base_name = output_name if output_name and output_name.strip() else Path(video_path).stem
            suffix = f"exported_{max_width}w_{quality}q"
            # GIFs are always saved permanently, others respect autosave setting.
            is_temp_save = export_format != "GIF" and not self.autosave_enabled
            output_path = self._generate_output_path(base_name, suffix, ext=ext, is_temp=is_temp_save)
            if export_format == "GIF": print(f"INFO: GIF format selected. Output will be saved to permanent folder: {output_path}")

            ffmpeg_cmd = [self.ffmpeg_exe, "-y", "-i", video_path]
            if export_format == "MP4":
                crf = int(28 - (quality / 100) * 10)
                ffmpeg_cmd.extend(["-vf", f"scale='min({max_width},iw)':-2:flags=lanczos", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k"])
            elif export_format == "WebM":
                crf = int(35 - (quality / 100) * 10)
                ffmpeg_cmd.extend(["-vf", f"scale='min({max_width},iw)':-2:flags=lanczos", "-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0", "-c:a", "libopus", "-b:a", "96k"])
            elif export_format == "GIF":
                progress(0.2, desc="Generating GIF palette (Pass 1/2)...")
                palette_path = self.temp_dir / "palette.png"
                palette_cmd = [self.ffmpeg_exe, "-y", "-i", video_path, "-vf", f"scale='min({max_width},iw)':-2:flags=lanczos,palettegen", str(palette_path)]
                subprocess.run(palette_cmd, check=True, capture_output=True, text=True)
                ffmpeg_cmd.extend(["-i", str(palette_path), "-filter_complex", f"[0:v]scale='min({max_width},iw)':-2:flags=lanczos[v];[v][1:v]paletteuse", "-an"])
            ffmpeg_cmd.append(str(output_path))
            progress(0.6, desc=f"Encoding {export_format}...")
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            return str(output_path)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: FFmpeg failed during export to {export_format}.\nCmd: {' '.join(e.cmd)}\nStderr: {e.stderr}"); return video_path
        except Exception as e: print(f"Error during export: {e}\n{traceback.format_exc()}"); return video_path

    def process_pipeline(self, input_path, operations, params, progress=gr.Progress()):
        """Processes a single video through a pipeline of operations."""
        current_video_path = input_path
        messages = [f"🚀 Starting pipeline for '{Path(input_path).name}'..."]
        execution_order = ["Frame Adjust", "Video Loop", "Export"]
        for op_name in execution_order:
            if op_name in operations:
                messages.append(f"  -> Starting '{op_name}' step...")
                original_path = current_video_path
                if op_name == "Frame Adjust":
                    current_video_path = self.adjust_frames(current_video_path, **params["frame_adjust"], progress=progress)
                elif op_name == "Video Loop":
                    current_video_path = self.create_loop(current_video_path, **params["loop"], progress=progress)
                elif op_name == "Export":
                    current_video_path = self.export_video(current_video_path, **params["export"], progress=progress)
                if current_video_path == original_path:
                    messages.append(f"❌ Operation '{op_name}' failed. Aborting pipeline.")
                    return None, "\n".join(messages)
                else:
                    messages.append(f"  -> '{op_name}' step completed.")
        return current_video_path, "\n".join(messages)

    def process_batch(self, input_paths, operations, params, progress=gr.Progress()):
        """Processes a batch of videos through the pipeline."""
        total_videos, final_video_path = len(input_paths), None
        if total_videos == 0: return None, "No videos provided for batch processing."
        batch_messages = [f"🚀 Starting batch process for {total_videos} videos..."]
        batch_folder_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        for i, video_path in enumerate(input_paths):
            progress(i / total_videos, desc=f"Processing video {i+1}/{total_videos}: {os.path.basename(video_path)}")
            batch_messages.append(f"\n--- Video {i+1}/{total_videos}: {os.path.basename(video_path)} ---")
            temp_result_path, messages = self.process_pipeline(video_path, operations, params, progress)
            batch_messages.append(messages)
            if temp_result_path:
                temp_path = Path(temp_result_path)
                final_path = self.output_dir / batch_folder_name / temp_path.name
                os.makedirs(final_path.parent, exist_ok=True)
                final_video_path = self._copy_to_permanent_storage(temp_result_path, final_path)
                batch_messages.append(f"✅ Batch result saved to: {final_path}")
            else: batch_messages.append(f"❌ Pipeline failed for {os.path.basename(video_path)}. Skipping.")
        batch_messages.append("\n--- ✅ Batch processing complete. ---")
        return final_video_path, "\n".join(batch_messages)
