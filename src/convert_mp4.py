# PACKAGES #
from pydicom.pixel_data_handlers.util import convert_color_space
import pydicom
import numpy as np
from PIL import Image
import cv2
import glob
import os
from tqdm import tqdm

# VARIABLES #
# local output directory
DATA_DIR_PATH = '/Users/cindywang/PycharmProjects/pda_echo/data/raw/output/'
# o2 output directory
ODATA_DIR_PATH = '/n/scratch3/users/a/ab455/Addie_PDA/output/'
# o2 missing directory
MDATA_DIR_PATH = '/n/scratch3/users/a/ab455/Addie_PDA/missing redownloaded'
DIRECTORIES = [ODATA_DIR_PATH, MDATA_DIR_PATH]
# DIRECTORIES = [DATA_DIR_PATH]


# helper functions
def load_dicom(file_path):
    """ Load DICOM file """
    return pydicom.dcmread(file_path)


def extract_slice(dicom_data, time_frame, slice_index):
    """ Extract a specific slice, accommodating both 3D and 4D DICOM data """
    # Get the number of dimensions
    num_dimensions = len(dicom_data.pixel_array.shape)
    if num_dimensions == 4:
        # 4D data: format (time, x, y, z)
        return dicom_data.pixel_array[time_frame, :, :, :]
    elif num_dimensions == 3:
        # 3D data: format (x, y, z), assuming no time dimension
        # Ignore the time_frame parameter
        return dicom_data.pixel_array
    else:
        # Handle unexpected data dimensions
        raise ValueError("Unsupported DICOM data dimensions: {}".format(num_dimensions))


def convert_to_png(slice_data, output_file):
    """ Convert the slice to PNG format """
    """ Process the slice for video frame, assuming RGB data """
    # Initialize an empty array with the same shape as the input, but with float type
    normalized_slice = np.zeros_like(slice_data, dtype=np.float32)

    # Normalize each channel
    for c in range(3):  # Assuming slice_data is in the shape [height, width, channels]
        min_val = np.min(slice_data[:, :, c])
        max_val = np.max(slice_data[:, :, c])
        normalized_slice[:, :, c] = (slice_data[:, :, c] - min_val) / (max_val - min_val)

    # Scale to 0-255 and convert to uint8
    normalized_slice = (normalized_slice * 255).astype(np.uint8)

    # Save as PNG
    image = Image.fromarray(normalized_slice)
    image.save(output_file)


def process_slice(slice_data):
    """ Process the slice for video frame, assuming RGB data """
    # Initialize an empty array with the same shape as the input, but with float type
    normalized_slice = np.zeros_like(slice_data, dtype=np.float32)

    # Normalize each channel
    for c in range(3):  # Assuming slice_data is in the shape [height, width, channels]
        min_val = np.min(slice_data[:, :, c])
        max_val = np.max(slice_data[:, :, c])
        normalized_slice[:, :, c] = (slice_data[:, :, c] - min_val) / (max_val - min_val)

    # Scale to 0-255 and convert to uint8
    normalized_slice = (normalized_slice * 255).astype(np.uint8)

    return normalized_slice


def create_mp4_from_slices(dicom_data, output_video_path, fps=15):
    """ Create MP4 from slices, assuming RGB data """
    num_frames_de = dicom_data.get((0x0028, 0x0008), 1)
    num_frames = int(num_frames_de.value) if isinstance(num_frames_de, pydicom.dataelem.DataElement) else 1
    slice_index = 0  # or whichever slice index you need
    color_form = dicom_data.PhotometricInterpretation
    print(color_form)

    # Handle single frame case
    if num_frames == 1:
        img = extract_slice(dicom_data, 0, slice_index)
        if color_form == 'YBR_FULL_422':
            img = convert_color_space(img, 'YBR_FULL_422', 'RGB')
        output_img_path = os.path.splitext(output_video_path)[0] + '.png'
        convert_to_png(img, output_img_path)
    else:
        img = extract_slice(dicom_data, 0, slice_index)
        if color_form == 'YBR_FULL_422':
            img = convert_color_space(img, 'YBR_FULL_422', 'RGB')
        # Initialize video writer
        first_frame = process_slice(img)
        height, width, channels = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Write the first frame
        video.write(first_frame)

        # Process and append each frame
        for time_frame in range(1, num_frames):
            slice_data = extract_slice(dicom_data, time_frame, slice_index)
            if color_form == 'YBR_FULL_422':
                slice_data = convert_color_space(slice_data, 'YBR_FULL_422', 'RGB')
            frame = process_slice(slice_data)
            video.write(frame)

        video.release()


# ITERATION #
# # Missing, special file structure
# for patient_id in os.listdir(MDATA_DIR_PATH):
#     # setup
#     if patient_id.startswith('.'):  # skips any hidden files, including .DS_Store
#         continue
#     # navigating directories + saving path names
#     print("Patient ID:", patient_id)
#     patient_path = os.path.join(MDATA_DIR_PATH, patient_id)  # ~/output/4936258
#     if not os.path.isdir(patient_path):
#         continue
#     dcm_files = glob.glob(os.path.join(patient_path, '**/*.dcm'), recursive=True)
#     # creating mp4 directory to save
#     mp4_output_path = patient_path.replace('a/ab455', 'c/cw294') + '/mp4/'
#     try:
#         os.makedirs(mp4_output_path, exist_ok=True)
#         print(f"Directory '{mp4_output_path}' created successfully")
#     except OSError as error:
#         print(f"Error creating directory '{mp4_output_path}': {error}")
#
#     # iterating through files
#     for dcm_file in tqdm(dcm_files):
#         try:
#             dicom_data = load_dicom(dcm_file)
#             # check if the DICOM file contains necessary pixel data
#             if not hasattr(dicom_data, 'PixelData'):
#                 raise ValueError("No Pixel Data found in DICOM file.")
#
#             dcm_name = os.path.splitext(dcm_file.split(patient_path + '/')[1])[0]
#             output_video_path = mp4_output_path + dcm_name + ".mp4"
#             create_mp4_from_slices(dicom_data, output_video_path)
#
#         except Exception as e:
#             print(f"Skipping file {dcm_file}: {e}")
#             continue


# Output, special file structure
for patient_id in os.listdir(ODATA_DIR_PATH):
    # setup
    # skips hidden/unwanted directories
    if patient_id.startswith('.') or patient_id.startswith('S') or patient_id.startswith('R'):
        continue
    # navigating directories + saving path names
    print("Patient ID:", patient_id)
    patient_path = os.path.join(ODATA_DIR_PATH, patient_id)  # ~/output/4936258
    if not os.path.isdir(patient_path):
        continue
    dcm_files = glob.glob(os.path.join(patient_path, '**/*.dcm'), recursive=True)
    for first_sub in os.listdir(patient_path):
        if first_sub.startswith('.'):
            continue
        sub1 = os.path.join(patient_path, first_sub)
        # sub1 = ~/4936258/20160114 - 20160114.182503 - no description (182503.89)
    for sub_sub in os.listdir(sub1):
        if sub_sub.startswith('.'):
            continue
        subdir_path = os.path.join(sub1, sub_sub)
        # subdir_path = ~/20160114 - 20160114.182503 - no description (182503.89)/1 - no description (182503.90)
        # creating mp4 directory to save
        mp4_output_path = subdir_path.replace('a/ab455', 'c/cw294') + '/mp4/'
        try:
            os.makedirs(mp4_output_path, exist_ok=True)
            print(f"Directory '{mp4_output_path}' created successfully")
        except OSError as error:
            print(f"Error creating directory '{mp4_output_path}': {error}")
        # iterating through files
        for dcm_file in tqdm(dcm_files):
            try:
                dicom_data = load_dicom(dcm_file)
                # check if the DICOM file contains necessary pixel data
                if not hasattr(dicom_data, 'PixelData'):
                    raise ValueError("No Pixel Data found in DICOM file.")
                if subdir_path in dcm_file:
                    dcm_name = os.path.splitext(dcm_file.split(subdir_path + '/')[1])[0]
                    output_video_path = mp4_output_path + dcm_name + ".mp4"
                    create_mp4_from_slices(dicom_data, output_video_path)
                else:
                    continue
            except Exception as e:
                print(f"Skipping file {dcm_file}: {e}")
                continue
