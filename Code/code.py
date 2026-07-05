# Fixed: Better header cropping to exclude axis labels from area calculation
# Scaling: 30m x 10m image.

!pip install pillow numpy matplotlib scikit-image --quiet

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color, measure
from google.colab import files

SCAN_LENGTH_M = 30.0
DEPTH_M = 10.0

os.makedirs("overlays", exist_ok=True)

def auto_crop_header(arr_rgb, black_frac_thresh=0.45, v_thresh=0.25, max_scan_rows=150):
    """Enhanced header removal: scans until we hit actual subsurface data."""
    hsv = color.rgb2hsv(arr_rgb)
    v = hsv[:,:,2]
    h = hsv[:,:,0]
    s = hsv[:,:,1]

    height, width = v.shape
    scan_limit = min(max_scan_rows, height-1)
    start_row = 0

    for r in range(scan_limit):
        # Check for dark/axis band (black or low-saturation)
        black_frac = np.mean(v[r, :] < v_thresh)
        low_sat_frac = np.mean(s[r, :] < 0.15)

        # Also check if we're hitting colorful subsurface data
        colorful_frac = np.mean((s[r, :] > 0.30) & (v[r, :] > 0.30))

        # Stop cropping when we have substantial colorful content and low black/gray
        if colorful_frac > 0.50 and black_frac < black_frac_thresh:
            start_row = r
            break

    return arr_rgb[start_row:, :, :], start_row

def extract_main_blue_curve(rgb):
    """Find the largest blue connected component and extract curve per column."""
    hsv = color.rgb2hsv(rgb)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    mask_blue = (h >= 0.55) & (h <= 0.75) & (s > 0.35) & (v > 0.28)

    labeled = measure.label(mask_blue, connectivity=2)
    props = measure.regionprops(labeled)

    if not props:
        print("No blue curve found; defaulting to bottom row.")
        return np.full(rgb.shape[1], rgb.shape[0], dtype=int)

    # Largest blue component
    largest = max(props, key=lambda p: p.area)
    blue_coords = largest.coords
    height, width = rgb.shape[:2]
    curve_y = np.full(width, height, dtype=int)

    for col in range(width):
        col_pts = blue_coords[blue_coords[:,1] == col]
        if col_pts.size > 0:
            # Take the topmost blue pixel in this column
            curve_y[col] = col_pts[:,0].min()

    return curve_y

def mask_above_curve(rgb, curve_y):
    """Mask all pixels above the blue curve."""
    height, width = rgb.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    for col in range(width):
        y_stop = curve_y[col]
        if y_stop < height:
            mask[:y_stop, col] = True
    return mask

def process_image_blue_curve(path):
    img = Image.open(path).convert('RGB')
    arr_full = np.asarray(img) / 255.0

    # Enhanced header crop
    arr, header_crop = auto_crop_header(arr_full)
    h, w, _ = arr.shape

    m_per_px_x = SCAN_LENGTH_M / float(w)
    m_per_px_y = DEPTH_M / float(h)
    px_area = m_per_px_x * m_per_px_y

    # Extract blue curve
    curve_y = extract_main_blue_curve(arr)
    mask_top = mask_above_curve(arr, curve_y)

    top_area_m2 = mask_top.sum() * px_area
    total_area_m2 = h * w * px_area
    bottom_area_m2 = total_area_m2 - top_area_m2

    # Overlay
    overlay = arr.copy()
    overlay[mask_top] = [0.0, 1.0, 0.0]
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join("overlays", f"{base}_overlay_fixed.png")
    overlay_img.save(out_path)

    # Blue curve visualization
    linemark = arr.copy()
    for col in range(w):
        y = curve_y[col]
        if y < h:
            linemark[y, col] = [0.0, 0.4, 1.0]
    line_img = Image.fromarray((linemark * 255).astype(np.uint8))
    line_path = os.path.join("overlays", f"{base}_curve_fixed.png")
    line_img.save(line_path)

    return {
        "file": os.path.basename(path),
        "top_area": top_area_m2,
        "bottom_area": bottom_area_m2,
        "total_area": total_area_m2,
        "overlay_path": out_path,
        "line_path": line_path,
        "header_rows_cropped": header_crop,
        "height_px": h,
        "width_px": w
    }

# Upload and process
uploaded = files.upload()
image_paths = sorted(uploaded.keys())
results = [process_image_blue_curve(p) for p in image_paths]

print("-- Top Layer (Region above blue curve, header excluded) --\n")
print(f"{'Image':<30}{'Total (m²)':>14}{'Top (m²)':>14}{'Bottom (m²)':>16}{'Top %':>10}")
for r in results:
    pct = 100.0 * r["top_area"] / r["total_area"] if r["total_area"] else 0
    print(f"{r['file']:<30}{r['total_area']:>14.2f}{r['top_area']:>14.2f}{r['bottom_area']:>16.2f}{pct:>10.2f}")

# Visualizations
for r, p in zip(results, image_paths):
    arr_cropped = np.asarray(Image.open(p).convert('RGB')) / 255.0
    arr_cropped, _ = auto_crop_header(arr_cropped)
    overlay = np.asarray(Image.open(r["overlay_path"]).convert('RGB')) / 255.0
    line_img = np.asarray(Image.open(r["line_path"]).convert('RGB')) / 255.0

    plt.figure(figsize=(18,6))
    plt.suptitle(f"{r['file']} | Top: {r['top_area']:.2f} m² (Header cropped: {r['header_rows_cropped']} rows)")
    plt.subplot(1, 3, 1); plt.title("Cropped Original"); plt.imshow(arr_cropped); plt.axis('off')
    plt.subplot(1, 3, 2); plt.title("Top Layer Highlighted"); plt.imshow(overlay); plt.axis('off')
    plt.subplot(1, 3, 3); plt.title("Extracted Blue Curve"); plt.imshow(line_img); plt.axis('off')
    plt.tight_layout(); plt.show()

# Graphs
labels = [os.path.splitext(r["file"])[0] for r in results]
idx = np.arange(1, len(results)+1, dtype=float)
top = np.array([r["top_area"] for r in results], dtype=float)
tot = np.array([r["total_area"] for r in results], dtype=float)
pct = np.clip(100.0 * np.divide(top, tot, out=np.zeros_like(top), where=tot>0), 0, 100)

# Plot 1: Top % by image
fig, ax = plt.subplots(figsize=(9,5))
bars = ax.bar(idx, pct, color='#2ecc71', edgecolor='#145a32', alpha=0.95)
for i, b in enumerate(bars):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1.2, f"{pct[i]:.1f}%",
            ha='center', va='bottom', fontsize=10, color='#145a32')
mean_pct = pct.mean() if len(pct) else 0.0
ax.axhline(mean_pct, color='#1f77b4', linestyle='--', linewidth=2, label=f"Average = {mean_pct:.1f}%")
if len(idx) >= 2:
    z = np.polyfit(idx, pct, 1); pfit = np.poly1d(z)
    ax.plot(idx, pfit(idx), color='#e74c3c', linewidth=2.3, marker='o', label=f"Trend: {z[0]:+.2f}%/image")
ax.set_title("Top Layer as % of Total Area (by blue curve)")
ax.set_xlabel("Image Index"); ax.set_ylabel("Top Layer (%)")
ax.set_xticks(idx, labels); ax.set_ylim(0, 105); ax.legend(loc='upper right')
plt.tight_layout(); plt.show()

# Plot 2: Migration-susceptible area
fig, ax = plt.subplots(figsize=(10,5))
bars = ax.bar(idx, top, color='#2ecc71', edgecolor='#145a32', alpha=0.95)
for i, b in enumerate(bars):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+top.max()*0.02 if len(top) else 0.02,
            f"{top[i]:.2f} m²", ha='center', va='bottom', fontsize=10, color='#145a32')
mean_top = top.mean() if len(top) else 0.0
ax.axhline(mean_top, color='#1f77b4', linestyle='--', linewidth=2, label=f"Average = {mean_top:.2f} m²")
if len(idx) >= 2:
    z = np.polyfit(idx, top, 1); pfit = np.poly1d(z)
    ax.plot(idx, pfit(idx), color='#e74c3c', linewidth=2.3, marker='o', label=f"Trend: {z[0]:+.2f} m²/image")
ax.set_title("Migration-Susceptible Area (Top) by Image")
ax.set_xlabel("Image Index"); ax.set_ylabel("Area (m²)")
ax.set_xticks(idx, labels); ax.set_ylim(0, max(top)*1.25 if len(top) else 1)
ax.legend(loc='upper left'); plt.tight_layout(); plt.show()
