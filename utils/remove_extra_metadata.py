"""
TODO: add descriptions
"""

import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime
import piexif


def remove_extra_metadata(image_path):
    """
    Standardize the metadata of an image, keeping those useful for object detection/image retrieval.

    :param image_path: Path of the image to process.
    :return: None.
    """
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Extract existing metadata
            exif_dict = piexif.load(img.info["exif"])

            # Metadata to keep (example, this list might need to be modified)
            keep_tags = [
                "ImageWidth", "ImageLength", "BitsPerSample", "Compression",
                "PhotometricInterpretation", "ImageDescription", "Make", "Model",
                "Orientation", "SamplesPerPixel", "XResolution", "YResolution",
                "ResolutionUnit", "Software", "Artist"
            ]

            # Standardize the date
            standard_date = datetime(1970, 1, 1, 0, 0, 0).strftime("%Y:%m:%d %H:%M:%S")
            exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = standard_date
            exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = standard_date
            exif_dict["0th"][piexif.ImageIFD.DateTime] = standard_date

            # Remove GPS data
            exif_dict["GPS"] = {}

            # Remove tags that are not in the keep_tags list
            for ifd in ("0th", "Exif", "1st"):
                for tag in list(exif_dict[ifd].keys()):
                    if TAGS.get(tag) not in keep_tags:
                        del exif_dict[ifd][tag]

            # Convert the modified EXIF dictionary to bytes
            exif_bytes = piexif.dump(exif_dict)

            # Save the image with the modified metadata
            img.save(image_path, exif=exif_bytes)

        print(f"Metadata standardized for: {image_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
