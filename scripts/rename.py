import os
import shutil

# ===== CONFIG =====
SOURCE_DIR = r"data\images\closed_eye"      # source folder
DEST_DIR   = r"data\images"  # destination folder
PREFIX     = "eye_close"          # change as needed
EXTENSIONS = (".jpg", ".png", ".jpeg")
# ==================

os.makedirs(DEST_DIR, exist_ok=True)

count = 1

for file in sorted(os.listdir(SOURCE_DIR)):
    if file.lower().endswith(EXTENSIONS):
        new_name = f"{PREFIX}_{count:04d}.jpg"

        src_path = os.path.join(SOURCE_DIR, file)
        dst_path = os.path.join(DEST_DIR, new_name)

        shutil.copy(src_path, dst_path)
        count += 1

print(f"Done! {count-1} images copied & renamed.")
