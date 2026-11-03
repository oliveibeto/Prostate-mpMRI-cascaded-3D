import time
from pathlib import Path
import torch
from dataset_prostate158 import Prostate158Dataset

PROJECT = Path(__file__).resolve().parent
ds = Prostate158Dataset(dataset_dir=Path(r"C:\ProstateData\prostate158_train\train"), # set to your Prostate158 train folder
                        target_spacing=(1.0, 1.0, 1.0), target_shape=(32, 128, 128),
                        cache_data=False)

out = PROJECT / "cache"
out.mkdir(exist_ok=True)
print(len(ds), "cases to cache")

t0 = time.time()
for i in range(len(ds)):
    f = out / f"case_{i:03d}.pt"
    if f.exists():
        continue
    torch.save(ds[i], f)
    print(f"{i + 1}/{len(ds)} saved | elapsed {(time.time() - t0) / 60:.1f} min")

print("Cache complete:", out)
print("\nWeight files found:")
for p in PROJECT.rglob("*.pth"):
    print(" ", p)