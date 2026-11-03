import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from preprocess_engine import ProstatePreprocessingEngine


def get_prostate158_file_list(dataset_dir):
    """Discovers valid patient directories and their corresponding modality paths."""
    dataset_path = Path(dataset_dir)
    patient_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])

    file_list = []
    for p_dir in patient_dirs:
        # Check standard modality names
        t2_path = p_dir / "t2.nii" if (p_dir / "t2.nii").exists() else p_dir / "t2.nii.gz"
        adc_path = p_dir / "adc.nii" if (p_dir / "adc.nii").exists() else p_dir / "adc.nii.gz"
        dce_path = p_dir / "dwi.nii" if (p_dir / "dwi.nii").exists() else p_dir / "dwi.nii.gz"

        # Whole-gland anatomical mask
        label_path = p_dir / "t2_anatomy_reader1.nii" if (
                    p_dir / "t2_anatomy_reader1.nii").exists() else p_dir / "t2_anatomy_reader1.nii.gz"

        if t2_path.exists() and adc_path.exists() and dce_path.exists():
            case_dict = {
                "patient_id": p_dir.name,
                "t2": str(t2_path),
                "adc": str(adc_path),
                "dce": str(dce_path),
                "label": str(label_path) if label_path.exists() else None
            }
            file_list.append(case_dict)

    return file_list


class Prostate158Dataset(Dataset):
    def __init__(self, dataset_dir, target_spacing=(1.0, 1.0, 1.0), target_shape=(32, 128, 128), cache_data=False):
        self.file_list = get_prostate158_file_list(dataset_dir)
        self.engine = ProstatePreprocessingEngine(target_spacing=target_spacing, target_shape=target_shape)
        self.cache_data = cache_data
        self.cache = {}

        print(f" Initialized Prostate158Dataset with {len(self.file_list)} valid cases.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.cache_data and idx in self.cache:
            return self.cache[idx]

        case = self.file_list[idx]

        # Process case using our robust preprocessing engine
        processed = self.engine.process_case(
            t2_path=case["t2"],
            adc_path=case["adc"],
            dce_path=case["dce"],
            label_path=case["label"]
        )

        sample = {
            "image": processed["image"],  # Shape: [3, 32, 128, 128]
            "label": processed.get("label", torch.zeros(1, 32, 128, 128)),  # Shape: [1, 32, 128, 128]
            "patient_id": case["patient_id"]
        }

        if self.cache_data:
            self.cache[idx] = sample

        return sample