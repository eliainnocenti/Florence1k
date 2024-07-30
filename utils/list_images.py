"""
This script is used to extract metadata from images in a directory and save it to a CSV file.

Functions:
----------
1. get_where_from(file_path)
    Get the 'Where from' attribute of a file.

2. get_image_metadata(file_path)
    Get metadata from an image file.

3. process_directory(directory)
    Process a directory of images and extract metadata.

4. create_csv(data, output_path, wrap_length=50)
    Save data to a CSV file.

5. custom_sort_key(item)
    Custom sorting function for folder names.

Dependencies:
-------------
- os
- csv
- PIL.Image
- pillow_heif
- datetime
- mimetypes
- subprocess
- textwrap

Usage:
------
To run this script, ensure that the required libraries are installed and the data directory is correctly set.

Author:
-------
Elia Innocenti
"""

import os
import csv
from PIL import Image
import pillow_heif
from datetime import datetime
import mimetypes
import subprocess
import textwrap


def get_where_from(file_path):
    """
    Function to get the 'Where from' attribute of a file.

    :param file_path:
    :return:
    """
    try:
        # Use the mdls command to get the kMDItemWhereFroms attribute
        result = subprocess.run(['mdls', '-name', 'kMDItemWhereFroms', file_path], capture_output=True, text=True)
        output = result.stdout.strip()

        # Extract the link from the result
        if "kMDItemWhereFroms =" in output:
            links = output.split("=", 1)[1].strip()
            # Remove parentheses and quotes if present
            links = links.strip('()"')
            # If there are multiple links, take the first one # TODO: check
            return links.split(',')[0].strip().strip('"')

    except Exception as e:
        print(f"Error retrieving 'Where from' for {file_path}: {str(e)}")

    return None


def get_image_metadata(file_path):
    """
    Function to get metadata from an image file.

    :param file_path:
    :return:
    """
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            format = img.format
            mode = img.mode

            # Get the last modification date
            mod_time = os.path.getmtime(file_path)
            mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')

            # Calculate the orientation
            orientation = "Horizontal" if width >= height else "Vertical"

            # Get the DPI resolution if available
            dpi = img.info.get('dpi', (None, None))

            # Check for transparency
            has_transparency = 'transparency' in img.info or 'A' in img.mode

            # Get the original download link
            download_link = get_where_from(file_path)

            return { # TODO: change attributes order (?)
                'file_name': os.path.basename(file_path),
                'folder_name': os.path.basename(os.path.dirname(file_path)),
                'width': width,
                'height': height,
                'size_kb': os.path.getsize(file_path) / 1024,
                'file_type': format,
                'mod_date': mod_date,
                'orientation': orientation,
                'dpi_x': dpi[0],
                'dpi_y': dpi[1],
                'color_space': mode,
                'has_transparency': has_transparency,
                'download_link': download_link or "N/A"
            }

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


def process_directory(directory):
    """
    Function to process a directory of images and extract metadata.

    :param directory:
    :return:
    """
    image_data = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type and mime_type.startswith('image'):
                metadata = get_image_metadata(file_path)
                if metadata:
                    image_data.append(metadata)
    return image_data


def create_csv(data, output_path, wrap_length=50):
    """
    Function to save data to a CSV file.

    :param data:
    :param output_path:
    :param wrap_length:
    :return:
    """
    if not data:
        print("No data to save.")
        return

    keys = data[0].keys()

    with open(output_path, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)

    '''
    with open(output_path, 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)

        # Write header
        writer.writerow(keys)

        # Write data rows
        for row in data:
            wrapped_row = []
            for key in keys:
                value = str(row[key])
                if len(value) > wrap_length:
                    wrapped_value = textwrap.fill(value, width=wrap_length)
                    wrapped_value = wrapped_value.replace('\n', '\n ')  # Add space for alignment
                    wrapped_row.append(wrapped_value)
                else:
                    wrapped_row.append(value)
            writer.writerow(wrapped_row)
    '''

    print(f"Data saved to {output_path}")


def custom_sort_key(item):
    """
    Custom sorting function for folder names and file names.
    Sorts primarily by folder number, then by folder name, and finally by file name.

    :param item: Dictionary containing the folder name.
    :return: Tuple containing the sort key and the folder name
    """
    folder_name = item['folder_name']
    file_name = item['file_name']

    try:
        # Extract the number from the beginning of the folder name
        folder_number = int(folder_name.split('.')[0])

    except ValueError:
        # If we can't extract a number, use the entire name as the key
        folder_number = float('inf')

    return folder_number, folder_name, file_name
