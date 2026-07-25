# U-Da-Real-Bad-Apple 🍎

An interactive, portable Python tool that transforms your personal photos and videos into a Bad Apple silhouette recap video with original audio preservation.

---

## 🌟 Key Features

- 🎯 **4 Interactive Layout Modes:**
  1. **Exact Silhouette Contour Fitting (Recommended):** Segments every character silhouette contour per frame (`cv2.connectedComponentsWithStats`) and masks photos pixel-by-pixel to exact shape outlines.
  2. **Bounding Box Silhouette Grid:** Fits a dynamic photo grid inside the overall bounding rectangle of the active silhouette shape.
  3. **Single Image Silhouette Fitting:** Fits 1 photo/video frame into the active silhouette shape at a time, perfectly synced to cycle through your entire media collection **1x** across the video length.
  4. **Static Canvas Overlay:** Displays a fixed global grid across the entire canvas.

- 📐 **Aspect-Preserving Center Clipping:** Zero stretching or distortion (`crop_and_resize_tile`).
- ⚡ **Configurable Preload Quality:** Choose between **Full Original Quality** (uncapped for high-spec GPUs), 2K, 1K, or 512p Performance.
- 🎞️ **Media Support:** Accepts images (`.jpg`, `.png`, `.webp`, `.heic`) and videos (`.mp4`, `.mov`, `.avi`, `.mkv`) by extracting the first frame.
- 🎵 **Audio Preservation:** Automatically muxes original AAC audio from `badapple.mp4` into the output video.
- 📁 **Fully Portable:** Auto-detects `badapple.mp4` and `images/` directory in the same folder.
- 🏷️ **Dynamic Option Filenames:** Saves each run with option tags (e.g. `BA - 1080p - Full - 4-3 - Silhouette.mp4`).

---

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Emmanuel-Roy/U-Da-Real-Bad-Apple.git
   cd U-Da-Real-Bad-Apple
   ```

2. **Install requirements:**
   ```bash
   pip install opencv-python numpy pillow tqdm pillow-heif imageio-ffmpeg
   ```

3. **Add your media:**
   Place your personal photos or video clips into the `images/` folder.

4. **Run the script:**
   ```bash
   python badapple.py
   ```

5. **Select your options interactively:**
   - Choose Layout Mode (`1`, `2`, `3`, or `4`)
   - Choose Orientation (`4:3` Landscape or `9:16` Vertical TikTok/Reels)
   - Choose Resolution (`1080p`, `4K`, or `8K`)
   - Choose Preload Quality (`Full`, `2K`, `1K`, or `Performance`)

---

## 📄 License
MIT License
