"""Train your own detector for your daily objects (pen, perfume, powerbank, mouse, headset, airpods).

HOW IT WORKS
  1. For each object, hold it up alone. The script finds it, draws a box and
     saves frames + labels automatically (no manual box drawing).
  2. It also records 'background' frames with none of your objects, so the
     model learns what NOT to detect (this removes most false guesses).
  3. It trains YOLO11 on all classes together and saves daily_model.pt,
     which app.py loads automatically.

RUN (stop app.py first, it uses the same camera):
  python train_daily.py

KEYS:  SPACE = start/pause auto capture   N = next object   ESC = quit
"""

import os
import time
import shutil
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO, YOLOWorld

# name shown in the app : text prompt used only to LOCATE it while collecting
CLASSES = {
    "pen": "pen",
    "perfume": "perfume bottle",
    "powerbank": "power bank",
    "mouse": "computer mouse",
    "headset": "headphones",
    "airpods": "airpods case",
}
IMAGES_PER_CLASS = 200
BACKGROUND_IMAGES = 150
CAPTURE_INTERVAL = 0.3      # seconds between saved frames
EPOCHS = 60

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "daily_dataset" / "raw"
YOLO_DIR = BASE_DIR / "daily_dataset" / "yolo"
OUT_MODEL = BASE_DIR / "daily_model.pt"
WORLD_FILE = BASE_DIR / "yolov8m-world.pt"

if torch.cuda.is_available():
    DEVICE = 0
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
WORLD_DEVICE = "cpu" if DEVICE == "mps" else DEVICE


def best_box(world, frame):
    r = world.predict(frame, imgsz=640, conf=0.25, verbose=False, device=WORLD_DEVICE)[0]
    if len(r.boxes) == 0:
        return None, 0.0
    i = int(r.boxes.conf.argmax())
    return r.boxes.xyxy[i].cpu().numpy(), float(r.boxes.conf[i])


def collect(cam, world, name, prompt, target_n, class_id):
    """Collect images for one class. prompt=None means background (no labels)."""
    out = RAW_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    if prompt:
        world.set_classes([prompt])

    saved, auto, last = 0, False, 0.0
    while saved < target_n:
        ok, frame = cam.read()
        if not ok or frame is None:
            continue
        frame = cv2.flip(frame, 1)          # same mirror view as app.py
        h, w = frame.shape[:2]
        disp = frame.copy()

        box, conf = (None, 0.0)
        if prompt:
            box, conf = best_box(world, frame)
            if box is not None:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(disp, f"{conf:.2f}", (x1, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        now = time.time()
        if auto and now - last >= CAPTURE_INTERVAL:
            stem = f"{name}_{saved:04d}"
            if prompt is None:
                cv2.imwrite(str(out / f"{stem}.jpg"), frame)
                (out / f"{stem}.txt").write_text("")
                saved += 1
                last = now
            elif box is not None and conf >= 0.35:
                x1, y1, x2, y2 = box
                area = ((x2 - x1) * (y2 - y1)) / (w * h)
                if 0.01 <= area <= 0.8:
                    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                    bw, bh = (x2 - x1) / w, (y2 - y1) / h
                    cv2.imwrite(str(out / f"{stem}.jpg"), frame)
                    (out / f"{stem}.txt").write_text(
                        f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    saved += 1
                    last = now

        if prompt:
            msg = f"HOLD UP: {name.upper()} (alone)"
            tip = "Rotate it, move closer/farther, change hand + background"
        else:
            msg = "BACKGROUND: show desk/room WITHOUT your objects"
            tip = "Point camera around, include hands, bottles, phones, etc."
        cv2.rectangle(disp, (0, 0), (w, 70), (12, 12, 12), -1)
        cv2.putText(disp, f"{msg}   {saved}/{target_n}   {'CAPTURING' if auto else 'PAUSED'}",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(disp, f"{tip} | SPACE=start/pause  N=next  ESC=quit",
                    (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.imshow("Collect dataset", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            auto = not auto
        elif key in (ord('n'), ord('N')):
            break
        elif key == 27:
            raise KeyboardInterrupt
    return saved


def build_yolo_dataset(names):
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
    for split in ("train", "val"):
        (YOLO_DIR / "images" / split).mkdir(parents=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True)

    for folder in sorted(RAW_DIR.iterdir()):
        if not folder.is_dir():
            continue
        imgs = sorted(folder.glob("*.jpg"))
        if not imgs:
            continue
        n_val = max(1, int(len(imgs) * 0.15))
        # last frames -> val (consecutive frames look alike, so don't shuffle)
        for i, img in enumerate(imgs):
            split = "val" if i >= len(imgs) - n_val else "train"
            shutil.copy(img, YOLO_DIR / "images" / split / img.name)
            shutil.copy(img.with_suffix(".txt"), YOLO_DIR / "labels" / split / (img.stem + ".txt"))

    yaml = f"path: {YOLO_DIR.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
    for i, n in enumerate(names):
        yaml += f"  {i}: {n}\n"
    (BASE_DIR / "daily_data.yaml").write_text(yaml)


def main():
    names = list(CLASSES.keys())
    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    time.sleep(0.8)
    world = YOLOWorld(str(WORLD_FILE))

    try:
        for idx, (name, prompt) in enumerate(CLASSES.items()):
            print(f"[COLLECT] {name}")
            collect(cam, world, name, prompt, IMAGES_PER_CLASS, idx)
        print("[COLLECT] background")
        collect(cam, world, "background", None, BACKGROUND_IMAGES, -1)
    except KeyboardInterrupt:
        print("[INFO] Collection stopped early, training on what was saved.")
    finally:
        cam.release()
        cv2.destroyAllWindows()

    build_yolo_dataset(names)

    print("[TRAIN] Starting training...")
    model = YOLO("yolo11s.pt")      # downloads automatically if missing
    model.train(
        data=str(BASE_DIR / "daily_data.yaml"), epochs=EPOCHS, imgsz=640, batch=16,
        device=DEVICE, workers=2, name="daily_objects", exist_ok=True, patience=20,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    shutil.copy(best, OUT_MODEL)
    print(f"\n[DONE] Saved {OUT_MODEL}. Now run: python app.py")


if __name__ == "__main__":
    main()