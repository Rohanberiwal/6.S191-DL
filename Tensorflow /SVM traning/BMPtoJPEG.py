import os
from PIL import Image

def convert_bmp_to_jpeg(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # List all files in the input folder
    files = os.listdir(input_folder)
    
    for file in files:
        if file.lower().endswith('.bmp'):
            bmp_path = os.path.join(input_folder, file)
            # Open BMP file
            try:
                with Image.open(bmp_path) as img:
                    # Generate output JPEG file path
                    jpeg_path = os.path.join(output_folder, os.path.splitext(file)[0] + '.jpg')
                    # Convert and save as JPEG
                    img.convert('RGB').save(jpeg_path)
                    print(f"Converted {file} to {jpeg_path}")
            except Exception as e:
                print(f"Failed to convert {file}: {e}")

# Replace with your actual paths
input_folder = r"C:\Users\rohan\OneDrive\Desktop\Tester"
output_folder = directory = r"C:\Users\rohan\OneDrive\Desktop\Train_mitotic"

convert_bmp_to_jpeg(input_folder, output_folder)
