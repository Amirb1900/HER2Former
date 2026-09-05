"""
bci_dataset_audit.py

هدف: یک بررسی جامع و سیستماتیک از دیتاست BCI برای کشف:
    1) ساختار دقیق پوشه‌ها و شمارش فایل‌ها
    2) صحت pairing بین HE/IHC (درون هر split، با کلید (split, id) نه فقط id)
    3) صحت pairing بین test/HE و groundtruth/groundtruth
    4) برخورد عددی id بین split های مختلف (train/test/val) - آمار سیستماتیک PSNR/SSIM
       (تعمیم همون کاری که قبلاً دستی برای train-test انجام دادید، این‌بار برای
       همه‌ی جفت‌های split ممکن)
    5) تشخیص duplicate واقعی تصویر (نه فقط اسم مشابه) با perceptual hashing سبک
       در کل دیتاست - این چیزیه که تا الان چک نشده

اجرا در Kaggle:
    python bci_dataset_audit.py
    یا مستقیم در یک سلول نوت‌بوک paste کنید و DATASET_ROOT را تنظیم کنید.

وابستگی‌ها: numpy, pandas, pillow, scikit-image (معمولاً در Kaggle از قبل نصب‌اند)
"""

import os
import re
import itertools
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# =========================================================================
# تنظیمات
# =========================================================================
DATASET_ROOT = "/kaggle/input/datasets/amirb1900/resized-bci-512/512BCI_dataset"

FOLDERS = {
    "train_HE":  Path(DATASET_ROOT) / "train" / "train" / "HE",
    "train_IHC": Path(DATASET_ROOT) / "train" / "train" / "IHC",
    "val_HE":    Path(DATASET_ROOT) / "val" / "val" / "HE",
    "val_IHC":   Path(DATASET_ROOT) / "val" / "val" / "IHC",
    "test_HE":   Path(DATASET_ROOT) / "test" / "test" / "HE",
    "groundtruth": Path(DATASET_ROOT) / "groundtruth" / "groundtruth",
}

FNAME_PATTERN = re.compile(r"^(\d+)_(train|test|val)_(\d\+|0)\.png$")

# آستانه‌ی perceptual hash برای این‌که دو تصویر را "نزدیک به duplicate" در نظر بگیریم
# (فاصله‌ی Hamming بین hash های ۶۴ بیتی؛ ۰ = دقیقا یکسان، هرچه بیشتر یعنی متفاوت‌تر)
HASH_DUPLICATE_THRESHOLD = 5


def hr(title=""):
    print("\n" + "=" * 80)
    if title:
        print(title)
        print("=" * 80)


# =========================================================================
# بخش ۱ - ساختار پوشه‌ها و شمارش فایل‌ها
# =========================================================================
def section_directory_structure():
    hr("بخش ۱: ساختار پوشه‌ها و شمارش فایل‌ها")
    rows = []
    for key, folder in FOLDERS.items():
        if not folder.exists():
            print(f"[MISSING] {key}: {folder}")
            rows.append({"folder": key, "path": str(folder), "n_files": 0})
            continue
        files = list(folder.glob("*.png"))
        rows.append({"folder": key, "path": str(folder), "n_files": len(files)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# =========================================================================
# بخش ۲ - پارس کردن نام فایل‌ها + تشخیص duplicate واقعی در سطح os.listdir
# =========================================================================
def parse_filename(fname: str):
    m = FNAME_PATTERN.match(fname)
    if not m:
        return None
    seq_number, split, level = m.groups()
    return {"seq_number": int(seq_number), "split": split, "her2_level": level}


def section_filename_integrity():
    hr("بخش ۲: صحت نام‌گذاری فایل‌ها + duplicate واقعی در لیست دایرکتوری")

    all_records = {}  # key -> list of dicts
    for key, folder in FOLDERS.items():
        if not folder.exists():
            continue

        # os.listdir به‌جای glob تا خود موتور فایل‌سیستم رو دقیق ببینیم
        raw_names = os.listdir(folder)
        name_counts = Counter(raw_names)
        exact_dupes = {n: c for n, c in name_counts.items() if c > 1}

        print(f"\n[{key}] تعداد کل ورودی‌ها در os.listdir: {len(raw_names)}")
        print(f"[{key}] تعداد نام‌های یکتا: {len(name_counts)}")
        if exact_dupes:
            print(f"[{key}] ⚠️  نام فایل‌هایی که بیش از یک‌بار در لیست دایرکتوری ظاهر شدند:")
            for n, c in list(exact_dupes.items())[:20]:
                print(f"     {n}  (×{c})")
            if len(exact_dupes) > 20:
                print(f"     ... و {len(exact_dupes) - 20} مورد دیگر")
        else:
            print(f"[{key}] duplicate واقعی در نام‌گذاری os.listdir دیده نشد.")

        parsed_bad = []
        records = []
        for fname in name_counts:  # فقط یکتاها را پردازش کن
            parsed = parse_filename(fname)
            if parsed is None:
                parsed_bad.append(fname)
                continue
            parsed["filename"] = fname
            parsed["folder_key"] = key
            parsed["full_path"] = folder / fname
            records.append(parsed)

        if parsed_bad:
            print(f"[{key}] ⚠️  {len(parsed_bad)} فایل با الگوی نام‌گذاری مطابقت ندارد. نمونه:")
            for n in parsed_bad[:10]:
                print(f"     {n}")

        all_records[key] = records

    return all_records


# =========================================================================
# بخش ۳ - صحت pairing درون هر split (HE <-> IHC) با کلید (split, seq_number)
# =========================================================================
def section_pairing_check(all_records):
    hr("بخش ۳: صحت pairing HE <-> IHC درون هر split (کلید صحیح: (split, seq_number))")

    def to_key_set(records):
        return {(r["split"], r["seq_number"]): r["her2_level"] for r in records}

    for split_name, he_key, ihc_key in [("train", "train_HE", "train_IHC"),
                                          ("val", "val_HE", "val_IHC")]:
        if he_key not in all_records or ihc_key not in all_records:
            continue
        he_set = to_key_set(all_records[he_key])
        ihc_set = to_key_set(all_records[ihc_key])

        he_keys = set(he_set.keys())
        ihc_keys = set(ihc_set.keys())

        common = he_keys & ihc_keys
        he_only = he_keys - ihc_keys
        ihc_only = ihc_keys - he_keys

        print(f"\n[{split_name}] HE: {len(he_keys)}   IHC: {len(ihc_keys)}   "
              f"مشترک: {len(common)}   فقط‌HE: {len(he_only)}   فقط‌IHC: {len(ihc_only)}")

        # چک برچسب یکسان بودن برای جفت‌های مشترک
        label_mismatches = [(k, he_set[k], ihc_set[k]) for k in common if he_set[k] != ihc_set[k]]
        if label_mismatches:
            print(f"[{split_name}] ⚠️  {len(label_mismatches)} جفت HE/IHC با برچسب HER2 متفاوت! نمونه:")
            for k, l1, l2 in label_mismatches[:10]:
                print(f"     {k}: HE={l1}  IHC={l2}")
        else:
            print(f"[{split_name}] همه‌ی جفت‌های مشترک برچسب یکسان دارند. ✓")

        if he_only:
            print(f"[{split_name}] نمونه id های فقط در HE: {list(he_only)[:10]}")
        if ihc_only:
            print(f"[{split_name}] نمونه id های فقط در IHC: {list(ihc_only)[:10]}")


# =========================================================================
# بخش ۴ - صحت pairing test/HE <-> groundtruth
# =========================================================================
def section_groundtruth_check(all_records):
    hr("بخش ۴: صحت pairing test/HE <-> groundtruth/groundtruth")

    if "test_HE" not in all_records or "groundtruth" not in all_records:
        print("پوشه‌های لازم پیدا نشدند.")
        return

    test_set = {(r["split"], r["seq_number"]): r["her2_level"] for r in all_records["test_HE"]}
    gt_set = {(r["split"], r["seq_number"]): r["her2_level"] for r in all_records["groundtruth"]}

    common = set(test_set) & set(gt_set)
    test_only = set(test_set) - set(gt_set)
    gt_only = set(gt_set) - set(test_set)

    print(f"test/HE: {len(test_set)}   groundtruth: {len(gt_set)}   "
          f"مشترک: {len(common)}   فقط‌test: {len(test_only)}   فقط‌gt: {len(gt_only)}")

    label_mismatches = [(k, test_set[k], gt_set[k]) for k in common if test_set[k] != gt_set[k]]
    if label_mismatches:
        print(f"⚠️  {len(label_mismatches)} جفت با برچسب متفاوت بین test/HE و groundtruth! نمونه:")
        for k, l1, l2 in label_mismatches[:10]:
            print(f"   {k}: test={l1}  groundtruth={l2}")
    else:
        print("همه‌ی جفت‌های مشترک برچسب یکسان دارند. ✓")

    if len(common) == len(test_set) == len(gt_set):
        print("✅ pairing کامل و یک‌به‌یک بین test/HE و groundtruth تایید شد.")


# =========================================================================
# بخش ۵ - آمار سیستماتیک PSNR/SSIM برای برخورد عددی id بین split های مختلف
# =========================================================================
def load_gray_resized(path: Path, size=(256, 256)):
    img = Image.open(path).convert("L").resize(size)
    return np.asarray(img).astype(np.float64)


def section_cross_split_collision_stats(all_records):
    hr("بخش ۵: آمار سیستماتیک PSNR/SSIM برای برخورد عددی id بین split های مختلف")
    print("(تعمیم همان تحلیلی که قبلاً برای train-test انجام دادید، این‌بار برای همه‌ی جفت‌های split)")

    split_sources = {
        "train": all_records.get("train_HE", []),
        "val": all_records.get("val_HE", []),
        "test": all_records.get("test_HE", []),
    }

    split_id_map = {}
    for split_name, records in split_sources.items():
        split_id_map[split_name] = {r["seq_number"]: r for r in records}

    for s1, s2 in itertools.combinations(split_id_map.keys(), 2):
        ids1 = set(split_id_map[s1].keys())
        ids2 = set(split_id_map[s2].keys())
        common_ids = ids1 & ids2

        print(f"\n--- {s1} vs {s2}: {len(common_ids)} برخورد عددی id از "
              f"{len(ids1)}×{len(ids2)} ممکن ---")

        if not common_ids:
            continue

        results = []
        for cid in common_ids:
            r1 = split_id_map[s1][cid]
            r2 = split_id_map[s2][cid]
            try:
                img1 = load_gray_resized(r1["full_path"])
                img2 = load_gray_resized(r2["full_path"])
                p = psnr(img1, img2, data_range=255)
                s_val = ssim(img1, img2, data_range=255)
                results.append({
                    "id": cid,
                    f"{s1}_file": r1["filename"],
                    f"{s2}_file": r2["filename"],
                    f"{s1}_label": r1["her2_level"],
                    f"{s2}_label": r2["her2_level"],
                    "psnr": p,
                    "ssim": s_val,
                })
            except Exception as e:
                print(f"[WARN] خطا در مقایسه‌ی id={cid}: {e}")

        if not results:
            continue

        df = pd.DataFrame(results)
        print(df[["psnr", "ssim"]].describe())

        # نمونه‌های مشکوک: شباهت بالا (احتمال duplicate واقعی، نه فقط برخورد تصادفی)
        suspicious = df[df["ssim"] > 0.5].sort_values("ssim", ascending=False)
        if len(suspicious) > 0:
            print(f"\n⚠️  {len(suspicious)} جفت با SSIM > 0.5 (نیازمند بررسی دستی):")
            print(suspicious.head(15).to_string(index=False))
        else:
            print("\nهیچ جفتی با SSIM > 0.5 دیده نشد -- مطابق انتظار برای برخورد تصادفی id.")


# =========================================================================
# بخش ۶ - تشخیص duplicate واقعی تصویر در کل دیتاست با perceptual hash سبک
# =========================================================================
def average_hash(path: Path, hash_size=8):
    """
    average hash ساده (بدون نیاز به پکیج اضافه مثل imagehash):
    تصویر را کوچک و grayscale می‌کند، میانگین روشنایی را حساب می‌کند،
    و یک بردار بیتی برمی‌گرداند (پیکسل روشن‌تر از میانگین = ۱).
    """
    img = Image.open(path).convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float64)
    avg = arr.mean()
    bits = (arr > avg).flatten()
    return bits


def hamming_distance(bits1, bits2):
    return int(np.count_nonzero(bits1 != bits2))


def section_full_dataset_duplicate_scan(all_records):
    hr("بخش ۶: اسکن کل دیتاست برای duplicate واقعی تصویر (perceptual hash)")
    print("این بخش صرفاً بر اساس نام فایل کار نمی‌کند - محتوای واقعی تصویر را می‌سنجد.")
    print(f"آستانه‌ی فاصله‌ی Hamming برای duplicate: {HASH_DUPLICATE_THRESHOLD} (از ۶۴ بیت)\n")

    all_items = []
    for key, records in all_records.items():
        for r in records:
            all_items.append((key, r["filename"], r["full_path"]))

    print(f"در حال محاسبه‌ی hash برای {len(all_items)} تصویر ... (ممکن است چند دقیقه طول بکشد)")

    hash_buckets = defaultdict(list)  # hash فشرده (برای bucketing تقریبی) -> لیست index ها
    hashes = []

    for i, (key, fname, path) in enumerate(all_items):
        try:
            h = average_hash(path)
        except Exception as e:
            print(f"[WARN] خطا در {key}/{fname}: {e}")
            h = None
        hashes.append(h)
        if h is not None:
            # bucketing تقریبی: ۸ بیت اول به‌عنوان کلید grouping اولیه
            bucket_key = tuple(h[:8])
            hash_buckets[bucket_key].append(i)

        if (i + 1) % 1000 == 0:
            print(f"  ... {i + 1}/{len(all_items)} پردازش شد")

    print("\nدر حال مقایسه‌ی درون هر bucket برای یافتن duplicate های واقعی ...")
    duplicate_pairs = []
    for bucket_key, indices in hash_buckets.items():
        if len(indices) < 2:
            continue
        for i, j in itertools.combinations(indices, 2):
            if hashes[i] is None or hashes[j] is None:
                continue
            dist = hamming_distance(hashes[i], hashes[j])
            if dist <= HASH_DUPLICATE_THRESHOLD:
                key_i, fname_i, _ = all_items[i]
                key_j, fname_j, _ = all_items[j]
                duplicate_pairs.append({
                    "folder_1": key_i, "file_1": fname_i,
                    "folder_2": key_j, "file_2": fname_j,
                    "hamming_distance": dist,
                })

    if duplicate_pairs:
        df = pd.DataFrame(duplicate_pairs).sort_values("hamming_distance")
        print(f"\n⚠️  {len(df)} جفت duplicate/near-duplicate احتمالی پیدا شد:")
        print(df.head(50).to_string(index=False))
        df.to_csv("/kaggle/working/duplicate_candidates.csv", index=False)
        print("\n[SAVED] لیست کامل در /kaggle/working/duplicate_candidates.csv ذخیره شد.")
    else:
        print("\n✅ هیچ duplicate/near-duplicate واقعی‌ای در کل دیتاست پیدا نشد.")

    return duplicate_pairs


# =========================================================================
# اجرای کامل
# =========================================================================
def main():
    section_directory_structure()
    all_records = section_filename_integrity()
    section_pairing_check(all_records)
    section_groundtruth_check(all_records)
    section_cross_split_collision_stats(all_records)
    section_full_dataset_duplicate_scan(all_records)

    hr("پایان بررسی")
    print("لطفاً کل خروجی کنسول (یا حداقل خلاصه‌ی هر بخش) را برای تحلیل بعدی ارسال کنید.")


if __name__ == "__main__":
    main()
