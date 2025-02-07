import os
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image


def convert_dicom_to_image(dicom_path, output_path, voi_lut=True, fix_monochrome=True):

    dicom = pydicom.dcmread(dicom_path)


    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # Fix MONOCHROME1 (inversion) if necessary
    if fix_monochrome and hasattr(dicom,
                                  'PhotometricInterpretation') and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    # Normalize the data to the range 0–255
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)


    image = Image.fromarray(data)
    image.save(output_path)


def process_directory(input_dir, output_dir, output_format='png', voi_lut=True, fix_monochrome=True):

    for subdir, dirs, files in os.walk(input_dir):

        rel_path = os.path.relpath(subdir, input_dir)

        output_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)

        for file in files:
            # Check if the file has a typical DICOM extension (adjust extensions if needed)
            if file.lower().endswith(('.dcm', '.dicom')):
                dicom_path = os.path.join(subdir, file)
                base_name = os.path.splitext(file)[0]
                output_file = f"{base_name}.{output_format}"
                output_path = os.path.join(output_subdir, output_file)

                try:
                    convert_dicom_to_image(dicom_path, output_path, voi_lut=voi_lut, fix_monochrome=fix_monochrome)
                    print(f"Converted: {dicom_path} --> {output_path}")
                except Exception as e:
                    print(f"Error converting {dicom_path}: {e}")


if __name__ == "__main__":
    # Input Directory
    input_directory = "D:\ED\DCOM_converter\data\\00001"

    #Output Directory
    output_directory = "D:\ED\DCOM_converter\data\output_images"

    process_directory(input_directory, output_directory, output_format='png')
