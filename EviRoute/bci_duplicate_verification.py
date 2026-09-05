"""
bci_duplicate_verification.py

هدف: دو بررسی تکمیلی و دقیق‌تر بعد از یافته‌های bci_dataset_audit.py

بخش الف) بررسی ارزان و قطعی (بدون نیاز به hash): آیا واقعاً همه/بیشتر فایل‌های
          val از نظر نام‌گذاری، suffix "_train_" دارند؟ این خودش مستقل از هر
          hash ای، به‌صورت مستقیم می‌گوید val زیرمجموعه‌ی نام‌گذاری train است یا نه.

بخش ب) بررسی دقیق‌تر duplicate واقعی: از بین جفت‌هایی که average-hash ضعیف
          آن‌ها را "یکسان" تشخیص داد، فقط آن‌هایی که در سطح پیکسل هم واقعاً
          شبیه هستند (با SSIM در رزولوشن کامل) را نگه می‌داریم. این باعث حذف
          false positive های ناشی از hash ضعیف می‌شود، مخصوصاً برای جفت‌های
          HE-IHC که به‌طور بصری اصلاً نباید شبیه باشند.

ورودی: فایل duplicate_candidates.csv که از اسکریپت قبلی تولید شده.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim

DATASET_ROOT = "/kaggle/input/datasets/amirb1900/resized-bci-512/512BCI_dataset"
DUPLICATE_CSV = "/kaggle/working/duplicate_candidates.csv"

FOLDERS = {
    "train_HE":  Path(DATASET_ROOT) / "train" / "train" / "HE",
    "train_IHC": Path(DATASET_ROOT) / "train" / "train" / "IHC",
    "val_HE":    Path(DATASET_ROOT) / "val" / "val" / "HE",
    "val_IHC":   Path(DATASET_ROOT) / "val" / "val" / "IHC",
    "test_HE":   Path(DATASET_ROOT) / "test" / "test" / "HE",
    "groundtruth": Path(DATASET_ROOT) / "groundtruth" / "groundtruth",
}

FNAME_PATTERN = re.compile(r"^(\d+)_(train|test|val)_(\d\+|0)\.png$")


def hr(title=""):
    print("\n" + "=" * 80)
    if title:
        print(title)
        print("=" * 80)


# =========================================================================
# بخش الف: بررسی نسبت suffix "_train_" در فایل‌های پوشه‌ی val
# =========================================================================
def section_val_naming_check():
    hr("بخش الف: چند درصد فایل‌های val واقعاً suffix '_train_' دارند؟")

    for key in ["val_HE", "val_IHC"]:
        folder = FOLDERS[key]
        if not folder.exists():
            continue
        files = list(folder.glob("*.png"))
        total = len(files)

        tag_counts = {"train": 0, "val": 0, "test": 0, "other": 0}
        for f in files:
            m = FNAME_PATTERN.match(f.name)
            if m is None:
                tag_counts["other"] += 1
                continue
            tag = m.group(2)
            tag_counts[tag] += 1

        print(f"\n[{key}] از مجموع {total} فایل:")
        for tag, count in tag_counts.items():
            pct = 100 * count / total if total else 0
            print(f"    suffix='{tag}': {count}  ({pct:.1f}%)")

        if tag_counts["train"] > 0:
            print(f"    ⚠️  {tag_counts['train']} فایل در پوشه‌ی {key} suffix '_train_' دارند "
                  f"-> شواهد قوی که این‌ها از مجموعه‌ی train کپی/renumber شده‌اند.")


# =========================================================================
# بخش ب: تایید دقیق duplicate ها با SSIM واقعی (رزولوشن کامل)
# =========================================================================
def load_gray_full(path: Path):
    img = Image.open(path).convert("L")
    return np.asarray(img).astype(np.float64)


def resolve_path(folder_key: str, filename: str) -> Path:
    return FOLDERS[folder_key] / filename


def section_verify_duplicates_with_ssim(max_pairs_to_check=2000):
    hr("بخش ب: تایید دقیق duplicate ها با SSIM در رزولوشن کامل")

    try:
        df = pd.read_csv(DUPLICATE_CSV)
    except FileNotFoundError:
        print(f"[ERROR] فایل {DUPLICATE_CSV} پیدا نشد. اول اسکریپت قبلی را اجرا کنید.")
        return

    print(f"تعداد کاندیدای duplicate از hash: {len(df)}")

    # اولویت با جفت‌هایی که بین split های مختلف (train/val/test) هستند،
    # چون این‌ها از نظر ریسک leakage مهم‌ترند
    def cross_split(row):
        s1 = row["folder_1"].split("_")[0]
        s2 = row["folder_2"].split("_")[0]
        return s1 != s2

    df["is_cross_split"] = df.apply(cross_split, axis=1)
    df_sorted = df.sort_values("is_cross_split", ascending=False)

    if len(df_sorted) > max_pairs_to_check:
        print(f"[INFO] فقط {max_pairs_to_check} جفت اول (با اولویت cross-split) بررسی می‌شود.")
        df_sorted = df_sorted.head(max_pairs_to_check)

    verified_results = []
    for i, row in df_sorted.iterrows():
        try:
            p1 = resolve_path(row["folder_1"], row["file_1"])
            p2 = resolve_path(row["folder_2"], row["file_2"])
            img1 = load_gray_full(p1)
            img2 = load_gray_full(p2)

            if img1.shape != img2.shape:
                # اگر سایز متفاوت بود، resize به کوچک‌تر
                h = min(img1.shape[0], img2.shape[0])
                w = min(img1.shape[1], img2.shape[1])
                img1 = np.array(Image.fromarray(img1).resize((w, h)))
                img2 = np.array(Image.fromarray(img2).resize((w, h)))

            s_val = ssim(img1, img2, data_range=255)
            verified_results.append({**row.to_dict(), "real_ssim": s_val})
        except Exception as e:
            verified_results.append({**row.to_dict(), "real_ssim": None, "error": str(e)})

        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1} جفت بررسی شد")

    vdf = pd.DataFrame(verified_results)

    # آستانه‌ی واقعی duplicate: SSIM بالای ۰.۹ در رزولوشن کامل (خیلی سخت‌گیرانه‌تر از hash خام)
    real_duplicates = vdf[vdf["real_ssim"] > 0.9].sort_values("real_ssim", ascending=False)
    false_positives = vdf[vdf["real_ssim"] <= 0.9]

    print(f"\n[نتیجه] از {len(vdf)} کاندیدای hash-based:")
    print(f"    ✅ duplicate واقعی تایید‌شده (SSIM>0.9): {len(real_duplicates)}")
    print(f"    ❌ false positive (hash ضعیف بوده): {len(false_positives)}")

    if len(real_duplicates) > 0:
        print("\nduplicate های واقعی (اولویت با cross-split):")
        print(real_duplicates[["folder_1", "file_1", "folder_2", "file_2",
                                 "real_ssim", "is_cross_split"]].head(30).to_string(index=False))

        cross_split_real = real_duplicates[real_duplicates["is_cross_split"]]
        print(f"\n⚠️  از این‌ها، {len(cross_split_real)} مورد بین splitهای مختلف هستند "
              f"(train/val/test) -> این‌ها مستقیماً روی leakage اثر دارند.")

    vdf.to_csv("/kaggle/working/duplicate_verified.csv", index=False)
    print("\n[SAVED] نتیجه‌ی کامل تایید‌شده در /kaggle/working/duplicate_verified.csv")


def main():
    section_val_naming_check()
    section_verify_duplicates_with_ssim()


if __name__ == "__main__":
    main()
