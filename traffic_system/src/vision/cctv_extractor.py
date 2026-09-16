import cv2
import numpy as np
import h5py
import subprocess
from pathlib import Path
from typing import Tuple, Union

class CCTVExtractor:
    def __init__(self, target_size: Tuple[int, int] = (224, 224), fps_sample_rate: int = 1):
        """
        Initializes the CCTV extractor for Phase 1 of the visual pipeline.
        :param target_size: The resolution to resize extracted frames (W, H).
        :param fps_sample_rate: Number of frames to extract per second of video.
        """
        self.target_size = target_size
        self.fps_sample_rate = fps_sample_rate

        # Standard ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def convert_proprietary_to_mp4(self, input_path: Union[str, Path], output_path: Union[str, Path]) -> bool:
        """
        Fallback function using FFmpeg to convert proprietary CCTV formats (.dav, .h264) to .mp4.
        """
        input_path = str(input_path)
        output_path = str(output_path)
        print(f"Attempting to convert {input_path} to {output_path} using FFmpeg...")

        try:
            # -y overwrites without asking
            cmd = ['ffmpeg', '-y', '-i', input_path, '-c:v', 'libx264', '-crf', '23', '-preset', 'fast', '-c:a', 'aac', output_path]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Successfully converted to {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg conversion failed: {e.stderr.decode('utf-8')}")
            return False
        except FileNotFoundError:
            print("FFmpeg not found. Please ensure ffmpeg is installed and added to system PATH.")
            return False

    def process_video(self, video_path: Union[str, Path], output_h5_path: Union[str, Path]) -> None:
        """
        Extracts frames, normalizes them, and saves to an HDF5 (.h5) file.
        """
        video_path = str(video_path)
        output_h5_path = str(output_h5_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # If OpenCV can't open it, it might be proprietary.
            print(f"Cannot open {video_path}. It may be a proprietary format.")
            return

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps == 0 or np.isnan(original_fps):
            original_fps = 30.0  # Fallback guess

        frame_skip = max(1, int(round(original_fps / self.fps_sample_rate)))
        print(f"Video original FPS: {original_fps:.2f} | Extracting 1 frame every {frame_skip} frames.")

        frames_list = []
        frame_idx = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                # Resize
                resized = cv2.resize(frame, self.target_size)

                # OpenCV loads BGR, convert to RGB
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

                # Normalize to [0, 1] then apply ImageNet mean/std
                normalized = rgb_frame.astype(np.float32) / 255.0
                normalized = (normalized - self.mean) / self.std

                # Channels-first format for PyTorch: (C, H, W)
                normalized = np.transpose(normalized, (2, 0, 1))

                frames_list.append(normalized)
                extracted_count += 1

            frame_idx += 1

        cap.release()

        if len(frames_list) > 0:
            frames_array = np.stack(frames_list, axis=0) # Shape: (Num_Frames, 3, 224, 224)
            print(f"Extracted {extracted_count} frames. Saving to {output_h5_path}...")

            # Save to HDF5 optimized for batched sequential access during training
            with h5py.File(output_h5_path, 'w') as f:
                f.create_dataset('frames', data=frames_array, compression="gzip", chunks=True)
                f.attrs['fps_sampled'] = self.fps_sample_rate
                f.attrs['original_video'] = video_path
        else:
            print(f"No frames were extracted from {video_path}.")

if __name__ == "__main__":
    # Example local test usage
    print("CCTVExtractor initialized.")
