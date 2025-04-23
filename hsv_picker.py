import cv2
import numpy as np
import argparse

# --- Globals to hold state ---
ref_hsv = None

def on_mouse(event, x, y, flags, param):
    global ref_hsv
    # param is (img_hsv, original_width)
    img_hsv, orig_w = param
    # only pick if click is within the original-image region (left half)
    if event == cv2.EVENT_LBUTTONDOWN and x < orig_w:
        ref_hsv = img_hsv[y, x].astype(int)
        print(f"Reference HSV set to: H={ref_hsv[0]} S={ref_hsv[1]} V={ref_hsv[2]}")

def nothing(x):
    pass

def main():
    parser = argparse.ArgumentParser(description="HSV-based interactive picker")
    parser.add_argument("image", help="Path to input image")
    args = parser.parse_args()

    # Load image
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print("Error: could not load image:", args.image)
        return

    # Convert once to HSV
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Create window and trackbars
    cv2.namedWindow("Original + Mask", cv2.WINDOW_NORMAL)
    
    cv2.createTrackbar("ΔH", "Original + Mask", 10, 90, nothing)    # Hue tolerance (0–90)
    cv2.createTrackbar("ΔS", "Original + Mask", 50, 255, nothing)   # Sat tolerance (0–255)
    cv2.createTrackbar("ΔV", "Original + Mask", 50, 255, nothing)   # Val tolerance (0–255)

    # Attach mouse callback to the combined window.
    # Pass both the HSV image and the width of the original image.
    orig_w = img_bgr.shape[1]
    cv2.setMouseCallback("Original + Mask", on_mouse, (img_hsv, orig_w))

    while True:
        # Read tolerances
        dH = cv2.getTrackbarPos("ΔH", "Original + Mask")
        dS = cv2.getTrackbarPos("ΔS", "Original + Mask")
        dV = cv2.getTrackbarPos("ΔV", "Original + Mask")

        if ref_hsv is None:
            # No ref yet: show white mask
            mask = np.ones((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.uint8) * 255
        else:
            # Compute per-channel difference with wrap-around for Hue
            h = img_hsv[:, :, 0].astype(int)
            s = img_hsv[:, :, 1].astype(int)
            v = img_hsv[:, :, 2].astype(int)

            dh = np.abs(h - ref_hsv[0])
            dh = np.minimum(dh, 180 - dh)   # wrap-around

            ds = np.abs(s - ref_hsv[1])
            dv = np.abs(v - ref_hsv[2])

            # mask = True where within all tolerances
            within = (dh <= dH) & (ds <= dS) & (dv <= dV)

            # black = close → 0; white = far → 255
            mask = np.where(within, 0, 255).astype(np.uint8)

        # Stack side by side
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([img_bgr, mask_bgr])

        # now only one display
        cv2.imshow("Original + Mask", combined)

        # Quit on 'q' or ESC
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
