import argparse
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from rembg import remove

def rm_bg(img: Image.Image) -> Image.Image:
    return remove(img, model="u2netp")

def crop_face(img: Image.Image, sf: float = 1.5) -> Image.Image:
    np_img = np.array(img.convert("RGB"))
    cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    fd = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    res = fd.process(cv_img)

    if not res.detections:
        raise ValueError("No faces were detected in the image.")
    if len(res.detections) > 1:
        raise ValueError(f"{len(res.detections)} faces detected. Please use an image with only one person.")

    det = res.detections[0]
    bbox = det.location_data.relative_bounding_box
    im_h, im_w, _ = cv_img.shape

    x = int(bbox.xmin * im_w)
    y = int(bbox.ymin * im_h)
    w = int(bbox.width * im_w)
    h = int(bbox.height * im_h)

    cx, cy = x + w // 2, y + h // 2
    nw = int(w * sf)
    nh = int(h * sf)

    nx = cx - nw // 2
    ny = cy - nh // 2

    l = max(0, nx)
    t = max(0, ny)
    r = min(im_w, nx + nw)
    b = min(im_h, ny + nh)

    return img.crop((l, t, r, b))

def finalize_img(rgba_img: Image.Image) -> Image.Image:
    np_fg = np.array(rgba_img)
    mask = np_fg[:, :, 3] > 10

    if np.any(mask):
        fg_px = np_fg[mask]
        avg_clr = fg_px[:, :3].mean(axis=0)
        dark_clr = tuple((avg_clr / 2).astype(int))
    else:
        dark_clr = (64, 64, 64)

    bg = Image.new("RGB", rgba_img.size, dark_clr)
    bg.paste(rgba_img, (0, 0), rgba_img)

    final = bg.convert("L")
    final = ImageOps.autocontrast(final, cutoff=2)
    final = ImageEnhance.Contrast(final).enhance(1.4)
    final = ImageEnhance.Brightness(final).enhance(0.95)

    return final

def main():
    p = argparse.ArgumentParser()
    p.add_argument("in_path")
    p.add_argument("out_path")
    a = p.parse_args()

    try:
        orig_img = Image.open(a.in_path)
        no_bg_img = rm_bg(orig_img)
        cropped_img = crop_face(no_bg_img)
        final_img = finalize_img(cropped_img)
        final_img.save(a.out_path)
    except Exception as e:
        raise e

if __name__ == "__main__":
    main()
