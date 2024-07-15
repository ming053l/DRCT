import argparse
import cv2
import glob
import numpy as np
import os
import onnxruntime
import time
import math
from tqdm import tqdm
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model.onnx', help='Path to the ONNX model')
    parser.add_argument('--input', type=str, default='input', help='Input folder with images')
    parser.add_argument('--output', type=str, default='output', help='Output folder')
    parser.add_argument('--scale', type=int, default=4, help='Upscaling factor')
    parser.add_argument('--tile_size', type=int, default=512, help='Tile size for processing')
    parser.add_argument('--tile_pad', type=int, default=32, help='Padding around tiles')
    args = parser.parse_args()
 
    # Load the ONNX model with CUDA Execution Provider
    ort_session = onnxruntime.InferenceSession(args.model_path, providers=['CUDAExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
 
    # Create output folder if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
 
    # Process each image in the input folder
    for image_path in tqdm(glob.glob(os.path.join(args.input, '*')), desc="Processing images", unit="image"):
        # Load image and normalize
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        original_height, original_width = img.shape[:2]
 
        # Upscale image using tiling
        output_img = tile_process(img, ort_session, input_name, args.scale, args.tile_size, args.tile_pad)
 
        # Convert to uint8 and save the upscaled image
        output_img = (output_img * 255.0).round().astype(np.uint8)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
 
        # Construct output filename with suffix and .png extension
        filename, _ = os.path.splitext(os.path.basename(image_path))
        output_filename = f"{filename}_REAL_GAN_DRCT.png"
        cv2.imwrite(os.path.join(args.output, output_filename), output_img)
 
 
def tile_process(img, ort_session, input_name, scale, tile_size, tile_pad):
    """Processes the image in tiles to avoid OOM errors."""
    height, width = img.shape[:2]
    output_height = height * scale
    output_width = width * scale
    output_shape = (output_height, output_width, 3)
 
    # Start with black image
    output_img = np.zeros(output_shape, dtype=np.float32)
 
    # Calculate number of tiles
    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)
 
    # Loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # Extract tile from input image
            ofs_x = x * tile_size
            ofs_y = y * tile_size
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_size, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_size, height)
 
            # Input tile area on total image with padding
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)
 
            # Input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y
            tile_idx = y * tiles_x + x + 1
            input_tile = img[input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad, :]
 
            # Pad tile to be divisible by scaling factor
            input_tile = pad_image(input_tile, 16)
 
            # Convert to BGR, transpose to CHW, and add batch dimension
            input_tile = np.transpose(input_tile[:, :, [2, 1, 0]], (2, 0, 1))
            input_tile = np.expand_dims(input_tile, axis=0).astype(np.float16)
 
            # Run inference
            output_tile = ort_session.run(None, {input_name: input_tile})[0]
 
            # Post-process the output tile
            output_tile = np.clip(output_tile, 0, 1)
            output_tile = np.transpose(output_tile[0, :, :, :], (1, 2, 0))
 
            # Output tile area on total image
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale
 
            # Output tile area without padding
            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + input_tile_width * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + input_tile_height * scale
 
            # Put tile into output image
            output_img[output_start_y:output_end_y, output_start_x:output_end_x, :] = output_tile[
                output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile, :
            ]
 
            print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')
 
    return output_img
 
 
def pad_image(img, factor):
    """Pads the image to be divisible by the given factor using reflection padding."""
    height, width = img.shape[:2]
    pad_height = (factor - (height % factor)) % factor
    pad_width = (factor - (width % factor)) % factor
    return cv2.copyMakeBorder(img, 0, pad_height, 0, pad_width, cv2.BORDER_REFLECT_101)
 
 
if __name__ == '__main__':
    main()
