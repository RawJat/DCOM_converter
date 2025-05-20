import os
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict


def get_slice_metrics(dicom_data):
    """Calculate metrics with proper handling of extreme values"""
    # Rescale data to positive range for calculations if needed
    calc_data = dicom_data.astype(np.float32)

    # Mean intensity (use absolute value for better comparison)
    mean_intensity = np.mean(calc_data)

    # Standard deviation
    std_intensity = np.std(calc_data)

    # Proper entropy calculation (ensure positive probabilities)
    hist, _ = np.histogram(calc_data.flatten(), bins=50)
    total = hist.sum()
    if total > 0:  # Avoid division by zero
        hist = hist / total
        # Filter out zero probabilities to avoid log(0)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
    else:
        entropy = 0.0

    return {
        'mean': mean_intensity,
        'std': std_intensity,
        'entropy': entropy
    }


def is_informative_slice(metrics, mean_threshold=None, std_threshold=1, entropy_threshold=0.5):
    """Determine if a slice is informative based on calculated metrics"""
    # For negative value datasets, we care more about std and entropy
    return (metrics['std'] > std_threshold and metrics['entropy'] > entropy_threshold)


def convert_dicom_to_image(dicom_path, output_path=None, voi_lut=True, fix_monochrome=True):
    """Convert a DICOM file to an image with proper value handling"""
    dicom = pydicom.dcmread(dicom_path)

    # Get original pixel data
    data = dicom.pixel_array

    # Calculate metrics before any normalization
    metrics = get_slice_metrics(data)

    # Apply VOI LUT transformation for visualization
    if voi_lut:
        try:
            data_display = apply_voi_lut(data, dicom)
        except:
            data_display = data.copy()
    else:
        data_display = data.copy()

    # Fix MONOCHROME1 photometric interpretation
    if fix_monochrome and hasattr(dicom,
                                  'PhotometricInterpretation') and dicom.PhotometricInterpretation == "MONOCHROME1":
        data_display = np.amax(data_display) - data_display

    # Normalize for display
    data_norm = data_display.astype(np.float32)
    data_norm = data_norm - np.min(data_norm)
    max_val = np.max(data_norm)
    if max_val > 0:
        data_norm = data_norm / max_val
    data_norm = (data_norm * 255).astype(np.uint8)

    if output_path:
        image = Image.fromarray(data_norm)
        image.save(output_path)

    return data_norm, metrics


def process_directory(input_dir, output_dir, output_format='png', voi_lut=True, fix_monochrome=True,
                      std_threshold=1, entropy_threshold=0.5, plot_profiles=True,
                      analyze_only=False, keep_best_n=None):
    """Process directory structure and filter by slice quality"""
    intensity_profiles = defaultdict(lambda: defaultdict(list))
    slice_counts = {'total': 0, 'kept': 0}
    all_metrics = []

    for subdir, dirs, files in os.walk(input_dir):
        dicom_files = [f for f in files if f.lower().endswith(('.dcm', '.dicom'))]
        if not dicom_files:
            continue

        rel_path = os.path.relpath(subdir, input_dir)
        output_subdir = os.path.join(output_dir, rel_path)

        path_parts = rel_path.split(os.path.sep)
        patient_id = path_parts[0] if len(path_parts) > 0 else "unknown"
        modality = path_parts[1] if len(path_parts) > 1 else "unknown"

        # Process DICOM files and get metrics
        slice_metrics = []
        for file in dicom_files:
            dicom_path = os.path.join(subdir, file)
            try:
                slice_num = int(os.path.splitext(file)[0].split('-')[-1])
            except:
                slice_num = len(slice_metrics)

            try:
                data, metrics = convert_dicom_to_image(dicom_path, None, voi_lut, fix_monochrome)
                slice_metrics.append((slice_num, file, data, metrics, dicom_path))
                intensity_profiles[patient_id][modality].append(
                    (slice_num, metrics['mean'], metrics['std'], metrics['entropy']))
                all_metrics.append((metrics['mean'], metrics['std'], metrics['entropy']))

            except Exception as e:
                print(f"Error processing {dicom_path}: {e}")

        if not slice_metrics:
            continue

        # Create output directory if we're saving files
        if not analyze_only:
            os.makedirs(output_subdir, exist_ok=True)

        # Handle keep_best_n option
        if keep_best_n:
            # Score based on std and entropy (more robust than mean for this dataset)
            for i, (slice_num, file, data, metrics, dicom_path) in enumerate(slice_metrics):
                quality_score = metrics['std'] + (3 * metrics['entropy'])
                slice_metrics[i] = (slice_num, file, data, metrics, dicom_path, quality_score)

            # Sort by quality score in descending order
            slice_metrics.sort(key=lambda x: x[5], reverse=True)
            selected_metrics = slice_metrics[:min(keep_best_n, len(slice_metrics))]
            slice_counts['total'] += len(slice_metrics)
            slice_counts['kept'] += len(selected_metrics)

            # Save the selected slices
            if not analyze_only:
                for slice_num, file, data, metrics, dicom_path, _ in selected_metrics:
                    base_name = os.path.splitext(file)[0]
                    output_path = os.path.join(output_subdir, f"{base_name}.{output_format}")
                    image = Image.fromarray(data)
                    image.save(output_path)
                    print(f"Saved slice: {output_path} (std={metrics['std']:.1f}, entropy={metrics['entropy']:.2f})")
        else:
            # Use traditional thresholding
            slice_counts['total'] += len(slice_metrics)
            for item in slice_metrics:
                slice_num, file, data, metrics, dicom_path = item
                should_keep = is_informative_slice(metrics, None, std_threshold, entropy_threshold)

                if should_keep:
                    slice_counts['kept'] += 1
                    if not analyze_only:
                        base_name = os.path.splitext(file)[0]
                        output_path = os.path.join(output_subdir, f"{base_name}.{output_format}")
                        image = Image.fromarray(data)
                        image.save(output_path)
                        print(
                            f"Saved slice: {output_path} (std={metrics['std']:.1f}, entropy={metrics['entropy']:.2f})")

    # Calculate metric statistics for threshold tuning
    if all_metrics:
        stds = [m[1] for m in all_metrics]
        entropies = [m[2] for m in all_metrics]

        print("\nMetric Statistics:")
        print(f"Std deviation: min={min(stds):.1f}, median={np.median(stds):.1f}, max={max(stds):.1f}")
        print(f"Entropy: min={min(entropies):.2f}, median={np.median(entropies):.2f}, max={max(entropies):.2f}")

        print("\nSuggested thresholds based on dataset:")
        print(f"std_threshold={np.percentile(stds, 60):.1f}  # 60th percentile")
        print(f"entropy_threshold={np.percentile(entropies, 60):.2f}  # 60th percentile")

    # Plot intensity profiles if requested
    if plot_profiles:
        plot_intensity_profiles(intensity_profiles, output_directory)

    print(
        f"\nProcessing complete: Kept {slice_counts['kept']}/{slice_counts['total']} slices ({slice_counts['kept'] / max(1, slice_counts['total']) * 100:.1f}%)")
    return all_metrics


def plot_intensity_profiles(intensity_profiles, output_directory):
    """Plot intensity profiles for each patient/modality"""
    for patient_id, modalities in intensity_profiles.items():
        fig, axs = plt.subplots(len(modalities), 1, figsize=(12, 4 * len(modalities)))
        if len(modalities) == 1:
            axs = [axs]

        fig.suptitle(f"Slice Intensity Profiles - Patient {patient_id}")

        for i, (modality, slices) in enumerate(modalities.items()):
            ax = axs[i]

            # Sort slices by slice number
            slices.sort()
            slice_nums = [s[0] for s in slices]
            means = [s[1] for s in slices]
            stds = [s[2] for s in slices]
            entropies = [s[3] for s in slices]

            # Plot std deviation as primary indicator of slice informativeness
            ax.plot(slice_nums, stds, 'g-', label='Std Deviation')
            # Plot entropy on secondary y-axis
            ax2 = ax.twinx()
            ax2.plot(slice_nums, entropies, 'r-', label='Entropy')

            ax.set_title(f"Modality: {modality}")
            ax.set_xlabel("Slice Number")
            ax.set_ylabel("Standard Deviation", color='g')
            ax2.set_ylabel("Entropy", color='r')

            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()
        plot_dir = os.path.join(os.path.dirname(output_directory), "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"patient_{patient_id}_intensity_profile.png"))
        plt.close()


if __name__ == "__main__":
    input_directory = r"D:\ED\DCOM_converter\data\00001"
    output_directory = r"D:\ED\DCOM_converter\data\filtered_images"

    # For your dataset, using keep_best_n is likely the most reliable approach
    process_directory(
        input_directory,
        output_directory,
        output_format='png',
        keep_best_n=6,  # Keep top 6 slices per modality
        analyze_only=False,  # Set to True for analysis only
        plot_profiles=True
    )