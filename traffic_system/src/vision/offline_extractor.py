import os
import cv2
import torch
import subprocess
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm

def process_cctv_offline(root_dir: str, output_dir: str):
    """
    Phase 1: Offline Extraction (Decoupled from Training)
    Iterates through proprietary CCTV footage, converts to MP4, extracts 1 frame per 15-min window,
    normalizes, and saves as independent .pt files mapped by (Date, Camera_ID, Time_Bucket).
    """
    root_path = Path(root_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    mp4box_path = root_path / "MP4Box.exe"

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if not mp4box_path.exists():
        print(f"Warning: MP4Box.exe not found at {mp4box_path}. Assuming available in system PATH.")
        mp4box_cmd = "MP4Box"
    else:
        mp4box_cmd = str(mp4box_path)

    # Directory mapping: Root -> Date (Level 1) -> Hex Camera_ID (Level 2) -> dados (Level 3) -> files
    for date_folder in root_path.iterdir():
        if not date_folder.is_dir(): continue
        date_str = date_folder.name

        for cam_folder in date_folder.iterdir():
            if not cam_folder.is_dir(): continue
            cam_id = cam_folder.name

            dados_folder = cam_folder / "dados"
            if not dados_folder.exists(): continue

            print(f"Processing Date: {date_str} | Camera: {cam_id}")

            for raw_file in tqdm(list(dados_folder.iterdir())):
                if not raw_file.is_file() or raw_file.suffix == '.mp4':
                    continue

                temp_mp4 = raw_file.with_suffix('.temp.mp4')

                # 1. Mux proprietary file to .mp4 using MP4Box
                try:
                    subprocess.run([mp4box_cmd, "-add", str(raw_file), str(temp_mp4)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                except subprocess.CalledProcessError:
                    if temp_mp4.exists(): temp_mp4.unlink()
                    continue # Skip corrupted file

                # 2. Extract 1 frame per 15-minute window
                cap = cv2.VideoCapture(str(temp_mp4))
                if not cap.isOpened():
                    temp_mp4.unlink()
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS)
                fps = 30.0 if (fps == 0 or fps != fps) else fps
                frame_interval = int(fps * 900) # 15 mins * 60 secs

                current_frame = 0
                time_bucket = 0

                while True:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                    ret, frame = cap.read()
                    if not ret: break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    tensor_frame = transform(frame_rgb)

                    # 3. Save resulting tensor locally as .pt
                    # Format: Date_CameraID_TimeBucket.pt
                    pt_filename = f"{date_str}_{cam_id}_TB{time_bucket:04d}.pt"
                    pt_filepath = out_path / pt_filename

                    torch.save(tensor_frame, pt_filepath)

                    current_frame += frame_interval
                    time_bucket += 1

                cap.release()
                temp_mp4.unlink() # Cleanup temporary mp4

if __name__ == "__main__":
    SOURCE_ROOT = r"D:\MMDA CCTV FOOTAGE\REQ. PUP STUDENT"
    OUTPUT_ROOT = str(Path(__file__).parent.parent.parent / "data" / "processed" / "cctv_tensors")
    print(f"Starting Offline Extraction...\nSource: {SOURCE_ROOT}\nOutput: {OUTPUT_ROOT}")
    process_cctv_offline(SOURCE_ROOT, OUTPUT_ROOT)
