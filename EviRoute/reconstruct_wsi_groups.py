"""
reconstruct_wsi_groups.py  (v2 - مطابق با ساختار واقعی پوشه‌های دیتاست شما)

ساختار مورد انتظار (طبق گزارشی که فرستادید):
    data_root/
        train/train/HE/*.png
        train/train/IHC/*.png
        val/val/HE/*.png
        val/val/IHC/*.png
        test/test/HE/*.png
        groundtruth/groundtruth/*.png      <- IHC واقعی برای test (نه در پوشه‌ی test)

نکات مهم که در این نسخه اصلاح شده:
    1. باگ قبلی: matching فقط بر اساس عدد id بود -> باعث می‌شد train_00017 با
       test_00017 اشتباهی جفت شوند، چون شماره‌گذاری در هر split از صفر شروع می‌شود.
       اصلاح: کلید matching الان (split, seq_number) است، نه فقط seq_number.
    2. split های train / val / test هرکدام کاملاً مستقل و جدا از هم گروه‌بندی
       می‌شوند (چون هیچ ارتباط عددی معناداری بین‌شان وجود ندارد -
       این را با آزمایش PSNR/SSIM شما هم تایید کردیم: میانگین SSIM=0.146).

اصل طراحی همان قبلی است: محافظه‌کارانه بودن در مرزگذاری.
بهتر است دو WSI متفاوت را اشتباهی ادغام کنیم (بی‌خطر) تا یک WSI واحد را
اشتباهی بشکنیم (خطرناک -> باعث بازگشت leakage می‌شود).

خروجی: یک CSV با ستون‌های:
    filename, split, her2_level, seq_number, pseudo_wsi_id

استفاده:
    python reconstruct_wsi_groups.py --data_root /path/to/512BCI_dataset --out groups.csv
"""

import argparse
import re
import csv
from pathlib import Path

import numpy as np
from PIL import Image

FNAME_PATTERN = re.compile(r"^(\d+)_(train|test|val)_(\d\+|0)\.png$")


def parse_filename(path: Path):
    m = FNAME_PATTERN.match(path.name)
    if not m:
        return None
    seq_number, split, level = m.groups()
    return {
        "filename": path.name,
        "split": split,
        "her2_level": level,
        "seq_number": int(seq_number),
        "path": path,
    }


def lightweight_embedding(img_path: Path, thumb_size=64) -> np.ndarray:
    img = Image.open(img_path).convert("RGB").resize((thumb_size, thumb_size))
    arr = np.asarray(img).astype(np.float32) / 255.0

    hsv = np.asarray(img.convert("HSV")).astype(np.float32) / 255.0
    hist_h, _ = np.histogram(hsv[..., 0], bins=16, range=(0, 1), density=True)
    hist_s, _ = np.histogram(hsv[..., 1], bins=16, range=(0, 1), density=True)
    hist_v, _ = np.histogram(hsv[..., 2], bins=16, range=(0, 1), density=True)

    gray = arr.mean(axis=-1)
    gx = np.abs(np.diff(gray, axis=0)).mean()
    gy = np.abs(np.diff(gray, axis=1)).mean()

    emb = np.concatenate([hist_h, hist_s, hist_v, [gx, gy]])
    norm = np.linalg.norm(emb) + 1e-8
    return emb / norm


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def segment_into_pseudo_wsi(records, embeddings, sim_threshold=0.75):
    group_ids = [0]
    boundary_reasons = ["start"]
    current_id = 0

    for i in range(1, len(records)):
        label_changed = records[i]["her2_level"] != records[i - 1]["her2_level"]
        sim = cosine_sim(embeddings[i], embeddings[i - 1])

        if label_changed:
            current_id += 1
            boundary_reasons.append(
                f"label_change ({records[i-1]['her2_level']} -> {records[i]['her2_level']})"
            )
        elif sim < sim_threshold:
            current_id += 1
            boundary_reasons.append(f"similarity_drop (sim={sim:.3f})")
        else:
            boundary_reasons.append("")

        group_ids.append(current_id)

    return group_ids, boundary_reasons


def get_he_dir(data_root: Path, split: str) -> Path:
    if split == "train":
        return data_root / "train" / "train" / "HE"
    elif split == "val":
        return data_root / "val" / "val" / "HE"
    elif split == "test":
        return data_root / "test" / "test" / "HE"
    raise ValueError(split)


def process_split(data_root: Path, split: str, sim_threshold: float):
    he_dir = get_he_dir(data_root, split)
    if not he_dir.exists():
        print(f"[SKIP] پوشه پیدا نشد: {he_dir}")
        return []

    files = sorted(he_dir.glob("*.png"))
    records = []
    seen_ids = set()

    for f in files:
        parsed = parse_filename(f)
        if parsed is None:
            print(f"[WARN] نام فایل با الگو مطابقت ندارد: {f.name}")
            continue
        key = (parsed["split"], parsed["seq_number"])
        if key in seen_ids:
            continue
        seen_ids.add(key)
        records.append(parsed)

    records.sort(key=lambda r: r["seq_number"])

    print(f"[{split}] در حال استخراج embedding برای {len(records)} تصویر ...")
    embeddings = [lightweight_embedding(r["path"]) for r in records]

    group_ids, reasons = segment_into_pseudo_wsi(records, embeddings, sim_threshold)

    for r, gid in zip(records, group_ids):
        r["pseudo_wsi_id"] = f"{split}_g{gid:04d}"

    n_groups = len(set(group_ids))
    print(f"[{split}] تعداد pseudo-WSI شناسایی‌شده: {n_groups} "
          f"(کل واقعی دیتاست حدود ۵۱ WSI است — این عدد سرجمع سه split نیست، "
          f"بلکه فقط سهم همین split از کل ۵۱ واقعی است)")

    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("groups.csv"))
    ap.add_argument("--sim_threshold", type=float, default=0.75)
    args = ap.parse_args()

    all_records = []
    for split in ["train", "val", "test"]:
        all_records.extend(process_split(args.data_root, split, args.sim_threshold))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "split", "her2_level", "seq_number", "pseudo_wsi_id"])
        for r in all_records:
            writer.writerow([r["filename"], r["split"], r["her2_level"],
                              r["seq_number"], r["pseudo_wsi_id"]])

    total_groups = len(set(r["pseudo_wsi_id"] for r in all_records))
    print(f"\n[DONE] خروجی نوشته شد در: {args.out}")
    print(f"[SUMMARY] مجموع pseudo-WSI groups در کل سه split: {total_groups}")
    print("        (این عدد لزوماً نباید ۵۱ باشد، چون train/val/test مستقل گروه‌بندی شدند "
          "و ممکن است یک WSI واقعی بین چند split پخش شده باشد -- که برای هدف ما مشکلی ندارد، "
          "چون فقط از این groups برای ساخت k-fold درون هر split یا روی کل داده استفاده می‌کنیم)")
    print("\n⚠️  قدم بعدی الزامی: چند نمونه از مرزهای شناسایی‌شده را با چشم بررسی کنید.")


if __name__ == "__main__":
    main()
