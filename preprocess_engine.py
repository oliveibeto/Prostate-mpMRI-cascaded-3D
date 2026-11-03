import SimpleITK as sitk
import numpy as np
import torch
from pathlib import Path


def remap_labels(label_arr):
    """
    Safely maps any arbitrary or out-of-bounds label values (e.g., [0, 3])
    into clean, contiguous integer indices starting from 0 (e.g., [0, 1]).
    """
    unique_vals = np.unique(label_arr)
    if np.array_equal(unique_vals, np.arange(len(unique_vals))):
        return label_arr

    remapped_arr = np.zeros_like(label_arr, dtype=np.int64)
    for new_idx, old_val in enumerate(unique_vals):
        remapped_arr[label_arr == old_val] = new_idx

    return remapped_arr


class ProstatePreprocessingEngine:
    def __init__(self, target_spacing=(1.0, 1.0, 1.0), target_shape=(32, 128, 128)):
        self.target_spacing = target_spacing
        self.target_shape = target_shape  # (Depth, Height, Width)

    def resample_volume(self, image, is_label=False):
        """Resamples image to isotropic 1.0mm grid using NearestNeighbor for labels and BSpline for images."""
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()

        if original_spacing == self.target_spacing:
            return image

        target_size = [
            int(round(original_size[i] * original_spacing[i] / self.target_spacing[i]))
            for i in range(3)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
        resampler.SetOutputSpacing(self.target_spacing)
        resampler.SetSize(target_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())

        return resampler.Execute(image)

    def register_volume(self, fixed_image, moving_image):
        """Rigidly registers moving volume (e.g. ADC/DCE) onto fixed reference volume (T2W)."""
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(0.1)
        registration.SetInterpolator(sitk.sitkLinear)

        registration.SetOptimizerAsGradientDescent(
            learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6
        )
        registration.SetOptimizerScalesFromPhysicalShift()

        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        registration.SetInitialTransform(initial_transform, inPlace=False)

        final_transform = registration.Execute(fixed_image, moving_image)

        return sitk.Resample(
            moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID()
        )

    def normalize_intensity(self, array):
        """Applies Z-score intensity normalization per channel."""
        mean = np.mean(array)
        std = np.std(array)
        if std > 0:
            return (array - mean) / std
        return array - mean

    def crop_or_pad_3d(self, array, is_label=False):
        """Crops or pads 3D array to standard input size [D, H, W]."""
        d_target, h_target, w_target = self.target_shape
        d_curr, h_curr, w_curr = array.shape

        # Depth
        if d_curr < d_target:
            pad = d_target - d_curr
            array = np.pad(array, ((pad // 2, pad - pad // 2), (0, 0), (0, 0)), mode='constant', constant_values=0)
        elif d_curr > d_target:
            start = (d_curr - d_target) // 2
            array = array[start:start + d_target, :, :]

        # Height
        if h_curr < h_target:
            pad = h_target - h_curr
            array = np.pad(array, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)), mode='constant', constant_values=0)
        elif h_curr > h_target:
            start = (h_curr - h_target) // 2
            array = array[:, start:start + h_target, :]

        # Width
        if w_curr < w_target:
            pad = w_target - w_curr
            array = np.pad(array, ((0, 0), (0, 0), (pad // 2, pad - pad // 2)), mode='constant', constant_values=0)
        elif w_curr > w_target:
            start = (w_curr - w_target) // 2
            array = array[:, :, start:start + w_target]

        return array

    def process_case(self, t2_path, adc_path, dce_path, label_path=None):
        """
        Unified pipeline execution:
        Resamples, co-registers, normalizes, crops/pads images AND synchronizes ground truth labels.
        """
        t2_img = sitk.ReadImage(str(t2_path))
        adc_img = sitk.ReadImage(str(adc_path))
        dce_img = sitk.ReadImage(str(dce_path))

        # 1. Resample to isotropic 1.0mm spacing
        t2_res = self.resample_volume(t2_img, is_label=False)
        adc_res = self.resample_volume(adc_img, is_label=False)
        dce_res = self.resample_volume(dce_img, is_label=False)

        # 2. Rigid Co-registration of ADC and DCE to T2W reference
        adc_reg = self.register_volume(t2_res, adc_res)
        dce_reg = self.register_volume(t2_res, dce_res)

        # 3. Convert to numpy arrays
        t2_arr = sitk.GetArrayFromImage(t2_res)
        adc_arr = sitk.GetArrayFromImage(adc_reg)
        dce_arr = sitk.GetArrayFromImage(dce_reg)

        # 4. Z-Score Normalization
        t2_norm = self.normalize_intensity(t2_arr)
        adc_norm = self.normalize_intensity(adc_arr)
        dce_norm = self.normalize_intensity(dce_arr)

        # 5. Crop/Pad to target shape [D, H, W]
        t2_crop = self.crop_or_pad_3d(t2_norm, is_label=False)
        adc_crop = self.crop_or_pad_3d(adc_norm, is_label=False)
        dce_crop = self.crop_or_pad_3d(dce_norm, is_label=False)

        # 6. Stack into 4D image tensor [C=3, D, H, W]
        stacked_4d = np.stack([t2_crop, adc_crop, dce_crop], axis=0)
        image_tensor = torch.from_numpy(stacked_4d).float()

        result = {"image": image_tensor}

        # 7. Synchronously process label mask if provided
        if label_path is not None:
            label_img = sitk.ReadImage(str(label_path))
            label_res = self.resample_volume(label_img, is_label=True)
            label_arr = sitk.GetArrayFromImage(label_res)

            # Safely remap arbitrary/out-of-bounds label values to contiguous indices
            label_arr = remap_labels(label_arr)

            label_crop = self.crop_or_pad_3d(label_arr, is_label=True)

            # Add channel dimension: [1, D, H, W]
            label_tensor = torch.from_numpy(np.expand_dims(label_crop, axis=0)).float()
            result["label"] = label_tensor

        return result