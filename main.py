"""
preprocess_prostate158.py

Preprocessing pipeline for Prostate158:
resample -> register (ADC/DWI onto T2) -> normalize -> crop/pad around the prostate.

Expects each patient folder (inside INPUT_DIR) to contain:
  t2.nii, adc.nii, dwi.nii,
  t2_anatomy_reader1.nii, t2_tumor_reader1.nii,
  adc_tumor_reader1.nii, adc_tumor_reader2.nii

HOW TO USE
----------
1. pip install SimpleITK numpy
2. Edit INPUT_DIR and OUTPUT_DIR below.
3. TEST ON ONE PATIENT FIRST before running the full loop (see the
   `if __name__ == "__main__":` block at the bottom for how to do that) -
   registration parameters sometimes need tuning per dataset, and it's much
   faster to catch that on 1 patient than after 158 have already run.
4. Once one patient's output looks right (open the .nii.gz files in 3D
   Slicer to sanity-check the T2/ADC/DWI overlap and that masks line up),
   run the full script.
"""

import os
import SimpleITK as sitk
import numpy as np

# ---------------- CONFIG ----------------
INPUT_DIR = r"/path/to/Prostate158"
OUTPUT_DIR = r"/path/to/Prostate158_preprocessed"
TARGET_SPACING = (1.0, 1.0, 1.0)   # mm, isotropic
CROP_SIZE = (128, 128, 64)         # voxels (x, y, z) -> 128mm x 128mm x 64mm at 1mm spacing
MARGIN_MM = 15.0                   # kept here for reference / future tuning
# -----------------------------------------


def load(path):
    return sitk.ReadImage(path)


def resample(image, target_spacing=TARGET_SPACING, is_mask=False, reference=None):
    """Resample `image` onto a grid with `target_spacing`, or onto `reference`'s grid if given."""
    if reference is not None:
        target_spacing = reference.GetSpacing()
        size = reference.GetSize()
        origin = reference.GetOrigin()
        direction = reference.GetDirection()
    else:
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        size = [
            int(round(osz * ospc / tspc))
            for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
        ]
        origin = image.GetOrigin()
        direction = image.GetDirection()

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(size)
    resampler.SetOutputOrigin(origin)
    resampler.SetOutputDirection(direction)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    return resampler.Execute(image)


def register(fixed, moving):
    """Rigid registration of `moving` onto `fixed`. Returns a transform (apply to masks too)."""
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=200
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_f, moving_f, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    reg.SetInitialTransform(initial_transform, inPlace=False)
    return reg.Execute(fixed_f, moving_f)


def apply_transform(image, reference, transform, is_mask=False):
    interp = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    return sitk.Resample(image, reference, transform, interp, 0.0, image.GetPixelID())


def normalize(image, mask=None):
    """Z-score normalize using only voxels inside `mask` (or all nonzero voxels if no mask)."""
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    mask_arr = (sitk.GetArrayFromImage(mask) > 0) if mask is not None else (arr > 0)
    if not mask_arr.any():
        mask_arr = np.ones_like(arr, dtype=bool)
    mean, std = arr[mask_arr].mean(), arr[mask_arr].std()
    arr = (arr - mean) / (std + 1e-8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out


def crop_or_pad_around_mask(image, anatomy_mask, crop_size=CROP_SIZE):
    """Crop/pad `image` to a fixed voxel size, centered on the anatomy mask's bounding box."""
    mask_arr = sitk.GetArrayFromImage(anatomy_mask) > 0  # z, y, x order
    size = image.GetSize()  # x, y, z order

    if not mask_arr.any():
        center_index = [s // 2 for s in size]
    else:
        coords = np.argwhere(mask_arr)
        center_zyx = coords.mean(axis=0)
        center_index = [int(center_zyx[2]), int(center_zyx[1]), int(center_zyx[0])]

    start = [max(0, center_index[i] - crop_size[i] // 2) for i in range(3)]
    start = [min(start[i], max(0, size[i] - crop_size[i])) for i in range(3)]
    safe_size = [min(crop_size[i], size[i] - start[i]) for i in range(3)]

    roi = sitk.RegionOfInterestImageFilter()
    roi.SetSize(safe_size)
    roi.SetIndex(start)
    cropped = roi.Execute(image)

    pad_after = [crop_size[i] - safe_size[i] for i in range(3)]
    if any(p > 0 for p in pad_after):
        cropped = sitk.ConstantPad(cropped, [0, 0, 0], pad_after, 0)
    return cropped


def process_patient(patient_dir, output_dir):
    patient_id = os.path.basename(patient_dir.rstrip("/\\"))
    print(f"Processing {patient_id}...")

    t2 = load(os.path.join(patient_dir, "t2.nii"))
    adc = load(os.path.join(patient_dir, "adc.nii"))
    dwi = load(os.path.join(patient_dir, "dwi.nii"))
    t2_anatomy = load(os.path.join(patient_dir, "t2_anatomy_reader1.nii"))
    t2_tumor = load(os.path.join(patient_dir, "t2_tumor_reader1.nii"))
    adc_tumor1 = load(os.path.join(patient_dir, "adc_tumor_reader1.nii"))
    adc_tumor2 = load(os.path.join(patient_dir, "adc_tumor_reader2.nii"))

    # 1. Resample T2 (the reference) to 1mm spacing; bring its own masks along on the same grid
    t2_rs = resample(t2, is_mask=False)
    t2_anatomy_rs = resample(t2_anatomy, is_mask=True, reference=t2_rs)
    t2_tumor_rs = resample(t2_tumor, is_mask=True, reference=t2_rs)

    # 2. Register ADC and DWI onto the resampled T2; carry each's masks with the same transform
    adc_transform = register(t2_rs, adc)
    adc_reg = apply_transform(adc, t2_rs, adc_transform, is_mask=False)
    adc_tumor1_reg = apply_transform(adc_tumor1, t2_rs, adc_transform, is_mask=True)
    adc_tumor2_reg = apply_transform(adc_tumor2, t2_rs, adc_transform, is_mask=True)

    dwi_transform = register(t2_rs, dwi)
    dwi_reg = apply_transform(dwi, t2_rs, dwi_transform, is_mask=False)

    # 3. Normalize intensities (scans only, never masks) within the prostate region
    t2_norm = normalize(t2_rs, mask=t2_anatomy_rs)
    adc_norm = normalize(adc_reg, mask=t2_anatomy_rs)
    dwi_norm = normalize(dwi_reg, mask=t2_anatomy_rs)

    # 4. Crop/pad everything to a fixed size, centered on the prostate
    out = {
        "t2": crop_or_pad_around_mask(t2_norm, t2_anatomy_rs),
        "adc": crop_or_pad_around_mask(adc_norm, t2_anatomy_rs),
        "dwi": crop_or_pad_around_mask(dwi_norm, t2_anatomy_rs),
        "t2_anatomy": crop_or_pad_around_mask(t2_anatomy_rs, t2_anatomy_rs),
        "t2_tumor": crop_or_pad_around_mask(t2_tumor_rs, t2_anatomy_rs),
        "adc_tumor_reader1": crop_or_pad_around_mask(adc_tumor1_reg, t2_anatomy_rs),
        "adc_tumor_reader2": crop_or_pad_around_mask(adc_tumor2_reg, t2_anatomy_rs),
    }

    patient_out_dir = os.path.join(output_dir, patient_id)
    os.makedirs(patient_out_dir, exist_ok=True)
    for name, img in out.items():
        sitk.WriteImage(img, os.path.join(patient_out_dir, f"{name}.nii.gz"))

    print(f"  done -> {patient_out_dir}")


def main(limit=None):
    """Set `limit` to an integer (e.g. 1) to test on just the first N patients."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    patient_dirs = [
        os.path.join(INPUT_DIR, d) for d in sorted(os.listdir(INPUT_DIR))
        if os.path.isdir(os.path.join(INPUT_DIR, d))
    ]
    if limit:
        patient_dirs = patient_dirs[:limit]
    print(f"Found {len(patient_dirs)} patient folder(s) to process.")

    for pdir in patient_dirs:
        try:
            process_patient(pdir, OUTPUT_DIR)
        except Exception as e:
            print(f"  FAILED on {pdir}: {e}")

    print("All done.")


if __name__ == "__main__":
    # Test on 1 patient first: main(limit=1)
    # Once you've checked the output, run the full set: main()
    main(limit=1)