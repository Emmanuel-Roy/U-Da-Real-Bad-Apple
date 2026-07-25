# U da real bad apple 🍎

Script to bound your own images to the silhouette of "Bad Apple".

✨ 100% vibe coded.

---

## 🌟 Key Features

- 🎯 **4 Layout Modes:**
  1. **Exact Silhouette Contour Fitting (Recommended):** Segments every character shape per frame and clips photos to the exact silhouette outlines.
  2. **Bounding Box Silhouette Grid:** Fits a dynamic photo grid inside the silhouette bounding box.
  3. **Single Image Silhouette Fitting:** Fits 1 photo/video into the silhouette at a time, synced 1x over the whole video.
  4. **Static Canvas Overlay:** Fixed global grid across the screen.

- 📐 **No Distortion:** Aspect-preserving center clipping so your photos never look stretched.
- ⚡ **Configurable Quality:** Choose between Full Original Quality (uncapped), 2K, 1K, or Performance 512p.
- 🎞️ **Media Support:** Accepts photos (`.jpg`, `.png`, `.webp`, `.heic`) and videos (`.mp4`, `.mov`, `.avi`).
- 🎵 **Audio Synced:** Keeps original audio track from `badapple.mp4`.
- 📁 **Portable:** Drop your files into the `images/` folder and run.

---

## 🚀 Quick Start

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Emmanuel-Roy/U-Da-Real-Bad-Apple.git
   cd U-Da-Real-Bad-Apple
   ```

2. **Install dependencies:**
   ```bash
   pip install opencv-python numpy pillow tqdm pillow-heif imageio-ffmpeg
   ```

3. **Drop your photos/videos into `images/`**

4. **Run the script:**
   ```bash
   python badapple.py
   ```

---

## 📄 License
MIT License
