import os
import glob
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None

# Enable HEIC / HEIF support in Pillow
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# ==========================================
# CONFIGURATION & PORTABLE PATH DETECTOR
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_portable_photos_dir():
    candidates = ["images", "photos", "my_photos", "input_photos"]
    for c in candidates:
        p = os.path.join(SCRIPT_DIR, c)
        if os.path.isdir(p):
            return p
    # Default to 'images' in script directory
    return os.path.join(SCRIPT_DIR, "images")

def get_portable_video_path():
    candidates = ["bad_apple.mp4", "badapple.mp4"]
    for c in candidates:
        p = os.path.join(SCRIPT_DIR, c)
        if os.path.isfile(p):
            return p
    # Fallback to any .mp4 in script directory
    mp4_files = glob.glob(os.path.join(SCRIPT_DIR, "*.mp4"))
    for f in mp4_files:
        basename = os.path.basename(f).lower()
        if not basename.startswith("bad_apple_recap") and not basename.startswith("temp_silent"):
            return f
    return os.path.join(SCRIPT_DIR, "bad_apple.mp4")

INPUT_PHOTOS_DIR = get_portable_photos_dir()
BAD_APPLE_VIDEO  = get_portable_video_path()
OUTPUT_VIDEO     = os.path.join(SCRIPT_DIR, "bad_apple_recap_final.mp4")
TEMP_VIDEO_PATH  = os.path.join(SCRIPT_DIR, "temp_silent_render.mp4")

# 4:3 Aspect Ratio Output Resolutions (Normal Mode)
RES_NORMAL_1080P = (1440, 1080)
RES_NORMAL_4K    = (2880, 2160)
RES_NORMAL_8K    = (5760, 4320)

# 9:16 Aspect Ratio Output Resolutions (Vertical TikTok / Reels)
RES_VERTICAL_1080P = (1080, 1920)
RES_VERTICAL_4K    = (2160, 3840)
RES_VERTICAL_8K    = (4320, 7680)

def build_output_filename(fit_mode, is_vertical, target_dim, max_dim):
    """Generates a dynamic output filename containing all user configuration options."""
    if target_dim in (RES_NORMAL_1080P, RES_VERTICAL_1080P):
        res_str = "1080p"
    elif target_dim in (RES_NORMAL_4K, RES_VERTICAL_4K):
        res_str = "4K"
    elif target_dim in (RES_NORMAL_8K, RES_VERTICAL_8K):
        res_str = "8K"
    else:
        res_str = f"{target_dim[0]}x{target_dim[1]}"

    if max_dim is None:
        qual_str = "Full"
    elif max_dim == 2048:
        qual_str = "2K"
    elif max_dim == 512:
        qual_str = "Performance"
    else:
        qual_str = "1K"

    aspect_str = "9-16" if is_vertical else "4-3"

    mode_map = {
        "exact": "Silhouette",
        "bbox": "BBox",
        "single_image": "SingleImage",
        "static": "Static"
    }
    mode_str = mode_map.get(fit_mode, fit_mode.capitalize())

    filename = f"BA - {res_str} - {qual_str} - {aspect_str} - {mode_str}.mp4"
    return os.path.join(SCRIPT_DIR, filename)

# ==========================================
# HARDWARE DETECTOR
# ==========================================
def detect_best_device():
    if torch is not None and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"\n[Hardware Acceleration]: CUDA / ROCm Detected -> {device_name}")
        return torch.device("cuda")

    try:
        import torch_directml
        if torch_directml.is_available():
            dml_device = torch_directml.device()
            print(f"\n[Hardware Acceleration]: AMD GPU Detected via DirectML ({dml_device})")
            return dml_device
    except ImportError:
        pass

    print("\n[Hardware Acceleration]: Running on CPU.")
    return None


# ==========================================
# PHOTO PROCESSING & CHRONOLOGICAL SORTING
# ==========================================
def center_crop_square(img):
    """Automatically center-crops any photo (portrait or landscape) to a 1:1 square."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    min_dim = min(w, h)
    
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    return img.crop((left, top, left + min_dim, top + min_dim))


def get_photo_timestamp(path):
    try:
        with Image.open(path) as img:
            exif = img._getexif()
            if exif:
                date_str = exif.get(36867)  # DateTimeOriginal
                if date_str and isinstance(date_str, str):
                    try:
                        dt = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                        return dt.timestamp()
                    except ValueError:
                        pass
    except Exception:
        pass
    return os.path.getmtime(path)


def crop_and_resize_tile(photo_bgr, target_w, target_h):
    """Center-crops a photo to match target_w x target_h aspect ratio, avoiding stretching or distortion."""
    h, w = photo_bgr.shape[:2]
    if h <= 0 or w <= 0 or target_w <= 0 or target_h <= 0:
        return cv2.resize(photo_bgr, (max(1, target_w), max(1, target_h)))

    target_aspect = target_w / target_h
    current_aspect = w / h

    if current_aspect > target_aspect:
        new_w = int(round(h * target_aspect))
        left = max(0, (w - new_w) // 2)
        cropped = photo_bgr[:, left : left + new_w]
    else:
        new_h = int(round(w / target_aspect))
        top = max(0, (h - new_h) // 2)
        cropped = photo_bgr[top : top + new_h, :]

    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        cropped = photo_bgr

    return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def load_single_item(path, max_dim=1024):
    """Loads an image or video frame, preserves natural aspect ratio, optionally caps max dimension, and returns BGR NumPy array directly."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.mp4', '.mov', '.m4v', '.avi', '.mkv'):
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = ImageOps.exif_transpose(img).convert("RGB")
            if max_dim is not None and max(img.size) > max_dim:
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return None
    else:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            if max_dim is not None and max(img.size) > max_dim:
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def sort_photos_chronologically(folder):
    supported_exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.heic", "*.heif", "*.mp4", "*.mov", "*.m4v", "*.avi", "*.mkv")
    photo_paths = []
    for ext in supported_exts:
        photo_paths.extend(glob.glob(os.path.join(folder, ext)))
        photo_paths.extend(glob.glob(os.path.join(folder, ext.upper())))

    source_video_abs = os.path.abspath(BAD_APPLE_VIDEO)
    output_video_abs = os.path.abspath(OUTPUT_VIDEO)
    temp_video_abs   = os.path.abspath(TEMP_VIDEO_PATH)

    filtered_paths = []
    for p in set(photo_paths):
        p_abs = os.path.abspath(p)
        if p_abs not in (source_video_abs, output_video_abs, temp_video_abs):
            filtered_paths.append(p)

    if not filtered_paths:
        raise FileNotFoundError(f"No valid images or videos found in folder '{folder}'.")

    print(f"Loading and sorting {len(filtered_paths)} media items chronologically...")
    filtered_paths.sort(key=get_photo_timestamp)
    return filtered_paths


def preload_and_square_photos(photo_paths, max_dim=1024):
    dim_str = f"max {max_dim}x{max_dim} px" if max_dim is not None else "full original uncapped resolution"
    print(f"Preloading {len(photo_paths)} media items ({dim_str})...")
    images = []
    for path in tqdm(photo_paths):
        try:
            arr = load_single_item(path, max_dim=max_dim)
            if arr is not None:
                images.append(arr)
        except Exception:
            continue
    if not images:
        raise ValueError("No valid images or videos could be loaded.")
    return images


# ==========================================
# CONFIGURATION PROMPTS
# ==========================================
def get_user_configuration():
    print("=" * 65)
    print(" BAD APPLE! DYNAMIC ALL-PHOTO FITTER ")
    print("=" * 65)

    print("Select Photo Layout Mode:")
    print("  [1] Exact Silhouette Contour Fitting (Fits photo grid to exact pixel boundaries per shape) [Recommended]")
    print("  [2] Bounding Box Silhouette Grid     (Fits dynamic photo grid inside overall silhouette bounding box)")
    print("  [3] Single Image Silhouette Fitting  (Fits 1 photo into silhouette at a time, synced to cycle 1x over video)")
    print("  [4] Static Canvas Overlay            (Fixed global grid across whole screen)")
    
    fit_choice = input("\nChoose layout mode (1, 2, 3, or 4) [Default: 1]: ").strip()
    if fit_choice == "2":
        fit_mode = "bbox"
    elif fit_choice == "3":
        fit_mode = "single_image"
    elif fit_choice == "4":
        fit_mode = "static"
    else:
        fit_mode = "exact"
    
    print("\nSelect Canvas Orientation:")
    print("  [1] Normal Landscape (4:3)  - Standard monitor / YouTube")
    print("  [2] Vertical Mobile  (9:16) - TikTok / Instagram Reels / Shorts")
    
    orient_choice = input("\nEnter orientation (1 or 2) [Default: 2]: ").strip()
    is_vertical = (orient_choice != "1")

    if is_vertical:
        print("\n--- VERTICAL MODE (9:16) ---")
        print("Select Target Resolution:")
        print("  [1] 1080p Vertical (1080x1920) - Standard mobile export")
        print("  [2] 4K Vertical    (2160x3840) - Recommended high detail")
        print("  [3] 8K Vertical    (4320x7680) - Maximum master archive detail")
        
        res_choice = input("\nChoose resolution (1, 2, or 3) [Default: 2]: ").strip()
        if res_choice == "1":
            target_dim = RES_VERTICAL_1080P
        elif res_choice == "3":
            target_dim = RES_VERTICAL_8K
        else:
            target_dim = RES_VERTICAL_4K
    else:
        print("\n--- NORMAL MODE (4:3) ---")
        print("Select Target Resolution:")
        print("  [1] 1080p Landscape (1440x1080) - Standard HD")
        print("  [2] 4K Landscape    (2880x2160) - Recommended high detail")
        print("  [3] 8K Landscape    (5760x4320) - Ultra HD")
        
        res_choice = input("\nChoose resolution (1, 2, or 3) [Default: 2]: ").strip()
        if res_choice == "1":
            target_dim = RES_NORMAL_1080P
        elif res_choice == "3":
            target_dim = RES_NORMAL_8K
        else:
            target_dim = RES_NORMAL_4K

    print("\nSelect Image Quality / Preload Resolution:")
    print("  [1] Full Original Quality (Uncapped - Max detail, requires high RAM / 16GB+ VRAM)")
    print("  [2] High Quality 2K       (2048x2048 - Recommended for 4K/8K renders)")
    print("  [3] Standard 1K           (1024x1024 - Recommended for 1080p renders) [Default]")
    print("  [4] Fast Performance      (512x512   - Low RAM usage)")
    
    qual_choice = input("\nChoose image quality (1, 2, 3, or 4) [Default: 3]: ").strip()
    if qual_choice == "1":
        max_dim = None
    elif qual_choice == "2":
        max_dim = 2048
    elif qual_choice == "4":
        max_dim = 512
    else:
        max_dim = 1024

    return fit_mode, is_vertical, target_dim, max_dim


def compute_dynamic_grid(target_w, target_h, num_photos):
    """Computes square grid geometry (cols, rows) based on canvas aspect ratio."""
    aspect = target_w / target_h
    cols = int(np.ceil(np.sqrt(num_photos * aspect)))
    rows = int(np.ceil(num_photos / cols))
    
    tile_size = max(1, min(target_w // cols, target_h // rows))
    offset_x = (target_w - cols * tile_size) // 2
    offset_y = (target_h - rows * tile_size) // 2
    return cols, rows, tile_size, tile_size, offset_x, offset_y


def merge_audio_track(temp_video_path, audio_source_path, final_output_path):
    print("\n[Audio Processing] Muxing original audio track into recap video...")
    ffmpeg_exe = None
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        import shutil
        ffmpeg_exe = shutil.which("ffmpeg")

    if not ffmpeg_exe:
        print("[Warning] ffmpeg executable not found. Saving video without audio.")
        if os.path.exists(temp_video_path) and temp_video_path != final_output_path:
            if os.path.exists(final_output_path):
                os.remove(final_output_path)
            os.rename(temp_video_path, final_output_path)
        return

    import subprocess
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", temp_video_path,
        "-i", audio_source_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        final_output_path
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if os.path.exists(temp_video_path) and temp_video_path != final_output_path:
            os.remove(temp_video_path)
        print(f"Done! Final recap video with audio saved to '{final_output_path}'")
    except Exception as e:
        print(f"[Warning] Could not merge audio track ({e}). Output saved as silent video.")
        if os.path.exists(temp_video_path) and temp_video_path != final_output_path:
            if os.path.exists(final_output_path):
                os.remove(final_output_path)
            os.rename(temp_video_path, final_output_path)


# ==========================================
# VIDEO RENDERING LOOP
# ==========================================
def render_video():
    detect_best_device()

    cap = cv2.VideoCapture(BAD_APPLE_VIDEO)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file '{BAD_APPLE_VIDEO}'")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fit_mode, is_vertical, (target_w, target_h), max_dim = get_user_configuration()

    photo_paths = sort_photos_chronologically(INPUT_PHOTOS_DIR)
    bgr_images = preload_and_square_photos(photo_paths, max_dim=max_dim)
    num_photos = len(bgr_images)

    static_cols, static_rows, static_tw, static_th, static_offx, static_offy = compute_dynamic_grid(target_w, target_h, num_photos)
    
    mode_names = {
        "exact": "Exact Silhouette Contour Fitting",
        "bbox": "Bounding Box Silhouette Grid",
        "single_image": "Single Image Silhouette Fitting",
        "static": "Static Canvas Overlay"
    }

    output_video = build_output_filename(fit_mode, is_vertical, (target_w, target_h), max_dim)

    print(f"\n[Portable Mode] Operating in: {SCRIPT_DIR}")
    print(f"  Source Video: {os.path.basename(BAD_APPLE_VIDEO)}")
    print(f"  Photo Folder: {os.path.basename(INPUT_PHOTOS_DIR)} ({num_photos:,} items loaded)")
    print(f"  Layout Mode:  {mode_names.get(fit_mode, fit_mode)}")
    print(f"  Output Video: {os.path.basename(output_video)}")

    temp_video_path = TEMP_VIDEO_PATH
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (target_w, target_h))

    # Calculate centered placement for source video
    if is_vertical:
        vid_w = target_w
        vid_h = int(target_w * (3.0 / 4.0))  # Centered 4:3 inside 9:16 frame
        pad_top = (target_h - vid_h) // 2
    else:
        vid_w = target_w
        vid_h = target_h
        pad_top = 0

    print(f"\nRendering Video Canvas ({target_w}x{target_h}) @ {fps:.2f} FPS...")
    pbar = tqdm(total=total_frames)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_source_np = cv2.resize(gray, (vid_w, vid_h), interpolation=cv2.INTER_LINEAR)

        # Full canvas mask
        full_mask = np.full((target_h, target_w), 255, dtype=np.uint8)
        if is_vertical:
            full_mask[pad_top:pad_top + vid_h, 0:vid_w] = resized_source_np
        else:
            full_mask = resized_source_np

        mask_float = full_mask.astype(np.float32) / 255.0

        # Start with a clean white canvas
        canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)

        if fit_mode == "exact":
            binary_silhouette = (full_mask < 200).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_silhouette, connectivity=8)
            total_silhouette_pixels = np.sum(binary_silhouette)

            if total_silhouette_pixels > 0 and num_labels > 1:
                photo_idx = 0
                valid_components = []
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area >= 50:
                        valid_components.append((i, area))

                if valid_components:
                    valid_components.sort(key=lambda item: item[1], reverse=True)
                    
                    for i, area in valid_components:
                        x = stats[i, cv2.CC_STAT_LEFT]
                        y = stats[i, cv2.CC_STAT_TOP]
                        w = stats[i, cv2.CC_STAT_WIDTH]
                        h = stats[i, cv2.CC_STAT_HEIGHT]

                        comp_share = area / total_silhouette_pixels
                        comp_num_photos = max(1, int(round(num_photos * comp_share)))

                        aspect = w / h
                        cols = max(1, int(np.ceil(np.sqrt(comp_num_photos * aspect))))
                        rows = max(1, int(np.ceil(comp_num_photos / cols)))

                        tile_w = max(1, w // cols)
                        tile_h = max(1, h // rows)

                        for r in range(rows):
                            for c in range(cols):
                                y1 = y + r * tile_h
                                y2 = (y + (r + 1) * tile_h) if r < rows - 1 else (y + h)
                                x1 = x + c * tile_w
                                x2 = (x + (c + 1) * tile_w) if c < cols - 1 else (x + w)

                                th = y2 - y1
                                tw = x2 - x1
                                if th <= 0 or tw <= 0 or y1 >= target_h or x1 >= target_w:
                                    photo_idx += 1
                                    continue

                                cell_label_mask = (labels[y1:y2, x1:x2] == i)

                                if np.any(cell_label_mask):
                                    cell_alpha = mask_float[y1:y2, x1:x2]
                                    effective_alpha = np.where(cell_label_mask, cell_alpha[:th, :tw], 1.0)

                                    photo_bgr = bgr_images[photo_idx % num_photos]
                                    tile_resized = crop_and_resize_tile(photo_bgr, tw, th)

                                    alpha_3d = effective_alpha[:th, :tw, np.newaxis]
                                    blended = tile_resized.astype(np.float32) * (1.0 - alpha_3d) + 255.0 * alpha_3d

                                    current_canvas_tile = canvas[y1:y2, x1:x2]
                                    np.copyto(current_canvas_tile, blended.astype(np.uint8), where=cell_label_mask[:, :, np.newaxis])

                                photo_idx += 1

        elif fit_mode == "bbox":
            black_coords = np.argwhere(full_mask < 200)
            if len(black_coords) > 0:
                ymin, xmin = black_coords.min(axis=0)
                ymax, xmax = black_coords.max(axis=0)
                bbox_h = max(1, ymax - ymin + 1)
                bbox_w = max(1, xmax - xmin + 1)

                aspect = bbox_w / bbox_h
                cols = max(1, int(np.ceil(np.sqrt(num_photos * aspect))))
                rows = max(1, int(np.ceil(num_photos / cols)))

                tile_w = max(1, bbox_w // cols)
                tile_h = max(1, bbox_h // rows)

                photo_idx = 0
                for r in range(rows):
                    for c in range(cols):
                        y1 = ymin + r * tile_h
                        y2 = (ymin + (r + 1) * tile_h) if r < rows - 1 else (ymax + 1)
                        x1 = xmin + c * tile_w
                        x2 = (xmin + (c + 1) * tile_w) if c < cols - 1 else (xmax + 1)

                        th = y2 - y1
                        tw = x2 - x1
                        if th <= 0 or tw <= 0 or y1 >= target_h or x1 >= target_w:
                            photo_idx += 1
                            continue

                        cell_mask = mask_float[y1:y2, x1:x2]

                        if cell_mask.size > 0 and np.min(cell_mask) < 0.9:
                            photo_bgr = bgr_images[photo_idx % num_photos]
                            tile_resized = crop_and_resize_tile(photo_bgr, tw, th)

                            alpha = cell_mask[:th, :tw, np.newaxis]
                            blended = tile_resized.astype(np.float32) * (1.0 - alpha) + 255.0 * alpha
                            canvas[y1:y2, x1:x2] = blended.astype(np.uint8)

                        photo_idx += 1

        elif fit_mode == "single_image":
            black_coords = np.argwhere(full_mask < 200)
            if len(black_coords) > 0:
                ymin, xmin = black_coords.min(axis=0)
                ymax, xmax = black_coords.max(axis=0)
                bbox_h = max(1, ymax - ymin + 1)
                bbox_w = max(1, xmax - xmin + 1)

                # Sync photo progression so media items cycle EXACTLY ONCE over total_frames
                photo_idx = min(num_photos - 1, int((frame_count - 1) * num_photos / max(1, total_frames)))
                photo_bgr = bgr_images[photo_idx]

                tile_resized = crop_and_resize_tile(photo_bgr, bbox_w, bbox_h)
                cell_mask = mask_float[ymin:ymax+1, xmin:xmax+1]

                alpha = cell_mask[:bbox_h, :bbox_w, np.newaxis]
                blended = tile_resized.astype(np.float32) * (1.0 - alpha) + 255.0 * alpha
                canvas[ymin:ymin+bbox_h, xmin:xmin+bbox_w] = blended.astype(np.uint8)

        else:  # static
            photo_idx = 0
            for r in range(static_rows):
                for c in range(static_cols):
                    y1 = static_offy + r * static_th
                    y2 = min(static_offy + (r + 1) * static_th, target_h)
                    x1 = static_offx + c * static_tw
                    x2 = min(static_offx + (c + 1) * static_tw, target_w)

                    if y1 >= target_h or x1 >= target_w or y1 >= y2 or x1 >= x2:
                        photo_idx += 1
                        continue

                    cell_mask = mask_float[y1:y2, x1:x2]

                    if cell_mask.size > 0 and np.min(cell_mask) < 0.9:
                        photo_bgr = bgr_images[photo_idx % num_photos]
                        th_actual, tw_actual = y2 - y1, x2 - x1
                        
                        tile_resized = crop_and_resize_tile(photo_bgr, tw_actual, th_actual)
                        alpha = cell_mask[:th_actual, :tw_actual, np.newaxis]

                        blended_tile = tile_resized.astype(np.float32) * (1.0 - alpha) + 255.0 * alpha
                        canvas[y1:y2, x1:x2] = blended_tile.astype(np.uint8)

                    photo_idx += 1

        out.write(canvas)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    
    # Merge audio track from bad_apple.mp4
    merge_audio_track(temp_video_path, BAD_APPLE_VIDEO, output_video)

if __name__ == "__main__":
    render_video()