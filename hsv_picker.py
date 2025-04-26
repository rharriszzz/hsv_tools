import cv2
import numpy as np
import argparse
import os

# Usage: python hsv_picker.py ../beads/beads-photo-2.jpg

# --- Globals to hold state ---
clicks = []              # store two clicks
custom_mask = None       # mask built from line picker
merge_mode = False       # if True, OR new region into existing mask
zoom = 1.0
offset_x = 0
offset_y = 0
last_mouse = (0, 0)
MIN_ZOOM = 1.0
MAX_ZOOM = 8.0
IMAGE_PATH = None        # will hold the input image path

def on_mouse(event, x, y, flags, param):
    global clicks, custom_mask, last_mouse, merge_mode
    global zoom, offset_x, offset_y
    # param is (img_hsv, original_width, original_height)
    img_hsv, orig_w, orig_h = param
    # remember where the mouse is (display coords)
    last_mouse = (x, y)

    # two‐click line picker (only inside left/original image region)
    if event == cv2.EVENT_LBUTTONDOWN and x < orig_w:
        # map display x,y → image coords
        ix = int(offset_x + x / zoom)
        iy = int(offset_y + y / zoom)
        clicks.append((ix, iy))
        if len(clicks) == 1:
            print(f"First point at {clicks[0]}")
        else:
            p1, p2 = clicks
            # rasterize line
            line_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            cv2.line(line_mask, p1, p2, 1, 1)
            # collect distinct HSV along that line, cast to int to avoid uint8 overflow
            line_hsv = img_hsv[line_mask == 1].astype(int)

            # get tolerances
            dH = cv2.getTrackbarPos("ΔH", "Original + Mask")
            dS = cv2.getTrackbarPos("ΔS", "Original + Mask")
            dV = cv2.getTrackbarPos("ΔV", "Original + Mask")

            # instead of raw unique, bucket line pixels by tolerance
            buckets = []
            for (h0, s0, v0) in line_hsv.reshape(-1, 3):
                placed = False
                for i, (h1, s1, v1, cnt) in enumerate(buckets):
                    dh = min(abs(h0 - h1), 180 - abs(h0 - h1))
                    if dh <= dH and abs(s0 - s1) <= dS and abs(v0 - v1) <= dV:
                        # merge
                        buckets[i] = (h1, s1, v1, cnt+1)
                        placed = True
                        break
                if not placed:
                    buckets.append((h0,s0,v0,1))

            # now buckets holds (h,s,v,line_count) for each cluster
            total_match = np.zeros((orig_h, orig_w), bool)
            total_count = 0
            for (h_val,s_val,v_val,line_cnt) in buckets:
                # build mask with same tolerance
                dh = np.abs(img_hsv[:,:,0].astype(int)-int(h_val))
                dh = np.minimum(dh, 180-dh)
                ds = np.abs(img_hsv[:,:,1].astype(int)-int(s_val))
                dv = np.abs(img_hsv[:,:,2].astype(int)-int(v_val))
                m = (dh<=dH)&(ds<=dS)&(dv<=dV)
                cnt = int(m.sum())
                print(f"H={h_val} S={s_val} V={v_val}  count={cnt}")
                total_match |= m
                total_count += cnt

            print(f"Total matched pixels: {total_count}")

            # build new mask from this line
            new_mask = np.where(total_match, 0, 255).astype(np.uint8)
            if merge_mode and custom_mask is not None:
                # merge into existing mask
                old_sel = (custom_mask == 0)
                new_sel = (new_mask == 0)
                merged = old_sel | new_sel
                custom_mask = np.where(merged, 0, 255).astype(np.uint8)
            else:
                # replace mask
                custom_mask = new_mask
            clicks.clear()
            # leave merge_mode enabled so further clicks also merge
            # (reset only on explicit 'r' key)
            # merge_mode = False

def nothing(x):
    pass

def run_loop(img_bgr, img_hsv, orig_w, orig_h):
    """Interactive display loop (zoom + mask)."""
    global zoom, offset_x, offset_y, last_mouse, custom_mask, merge_mode
    global IMAGE_PATH
    while True:
        # exit cleanly if user closed the window
        if cv2.getWindowProperty("Original + Mask", cv2.WND_PROP_VISIBLE) < 1:
            break

        # handle zoom keys
        key = cv2.waitKey(30) & 0xFF
        # merge region into existing mask
        if key == ord('m'):
            merge_mode = True
            clicks.clear()
            print("Merge mode enabled: click two points to add line regions")
            continue
        # reset merge mode
        elif key == ord('r'):
            merge_mode = False
            print("Merge mode disabled")
            continue
        # save mask
        if key == ord('s'):
            # choose the mask to save (full resolution)
            mask_to_save = custom_mask if custom_mask is not None else \
                           np.full((orig_h, orig_w), 255, dtype=np.uint8)
            # prompt for suffix
            suffix = input("Enter filename suffix: ")
            base = os.path.basename(IMAGE_PATH)
            stem, ext = os.path.splitext(base)
            out_name = f"{stem}-{suffix}{ext}"
            cv2.imwrite(out_name, mask_to_save)
            print(f"Mask saved to {out_name}")
            continue
        if key in (ord('+'), ord('=')):
            # zoom in around mouse
            mx, my = last_mouse
            cx = offset_x + mx / zoom
            cy = offset_y + my / zoom
            zoom = min(zoom * 2, MAX_ZOOM)
            vw, vh = orig_w/zoom, orig_h/zoom
            offset_x = max(0, min(orig_w - vw, cx - vw/2))
            offset_y = max(0, min(orig_h - vh, cy - vh/2))
        elif key == ord('-'):
            # zoom out around mouse
            mx, my = last_mouse
            cx = offset_x + mx / zoom
            cy = offset_y + my / zoom
            zoom = max(zoom / 2, MIN_ZOOM)
            vw, vh = orig_w/zoom, orig_h/zoom
            offset_x = max(0, min(orig_w - vw, cx - vw/2))
            offset_y = max(0, min(orig_h - vh, cy - vh/2))
        elif key in (ord('q'), 27):
            break

        # compute current viewport size and center (so cx,cy always exist)
        vw = orig_w / zoom
        vh = orig_h / zoom
        cx = offset_x + vw/2
        cy = offset_y + vh/2

        # update viewport
        offset_x = max(0, min(orig_w - vw, cx - vw/2))
        offset_y = max(0, min(orig_h - vh, cy - vh/2))

        # choose mask
        if custom_mask is None:
            mask = np.full((orig_h, orig_w), 255, dtype=np.uint8)
        else:
            mask = custom_mask

        # crop & resize image & mask for display
        x0, y0 = int(offset_x), int(offset_y)
        x1, y1 = int(offset_x + vw), int(offset_y + vh)
        img_crop  = img_bgr[y0:y1, x0:x1]
        mask_crop = mask   [y0:y1, x0:x1]
        img_disp  = cv2.resize(img_crop,  (orig_w, orig_h),
                                interpolation=cv2.INTER_NEAREST)
        mask_bgr  = cv2.cvtColor(mask_crop, cv2.COLOR_GRAY2BGR)
        mask_disp = cv2.resize(mask_bgr,(orig_w, orig_h),
                                interpolation=cv2.INTER_NEAREST)
        combined  = np.hstack([img_disp, mask_disp])
        cv2.imshow("Original + Mask", combined)

def main():
    global zoom, offset_x, offset_y, last_mouse, custom_mask
    parser = argparse.ArgumentParser(description="HSV-based interactive picker")
    parser.add_argument("image", help="Path to input image")
    args = parser.parse_args()
    # remember input path for saving masks
    global IMAGE_PATH
    IMAGE_PATH = args.image

    # ---- Interaction summary ----
    print("HSV Picker Controls:")
    print("  • Left‐click two points on the left image → pick a line")
    print("    → reports each distinct HSV along that line and their full‐image counts")
    print("  • + or =      : zoom in")
    print("  • -           : zoom out")
    print("  • q  or Esc   : quit")
    print("  • s           : save current mask to file")
    print("  • m           : merge new line region into existing mask")
    print("  • r           : reset merge mode")
    print("------------------------------\n")

    # Load image
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print("Error: could not load image:", args.image)
        return

    # Convert once to HSV
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Create window and trackbars
    # resizable window that preserves square pixels
    cv2.namedWindow("Original + Mask",
                    cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    
    cv2.createTrackbar("ΔH", "Original + Mask", 10, 90, nothing)    # Hue tolerance (0–90)
    cv2.createTrackbar("ΔS", "Original + Mask", 50, 255, nothing)   # Sat tolerance (0–255)
    cv2.createTrackbar("ΔV", "Original + Mask", 50, 255, nothing)   # Val tolerance (0–255)

    # Attach mouse callback to the combined window.
    # Pass both the HSV image and the width of the original image.
    orig_h, orig_w = img_bgr.shape[:2]
    cv2.setMouseCallback("Original + Mask", on_mouse, (img_hsv, orig_w, orig_h))

    # run the interactive loop, catching Ctrl-C
    try:
        run_loop(img_bgr, img_hsv, orig_w, orig_h)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
