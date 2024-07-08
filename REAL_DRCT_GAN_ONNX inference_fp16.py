import argparse
import cv2
import glob
import numpy as np
import os
import onnxruntime
import time
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model.onnx', help='Path to ONNX model')
    parser.add_argument('--input', type=str, default='input', help='Input folder with images')
    parser.add_argument('--output', type=str, default='output', help='Output folder')
    parser.add_argument('--scale', type=int, default=4, help='Scaling factor')
    parser.add_argument('--tile', type=int, default=800, help='Tile size')
    parser.add_argument('--tile_overlap', type=int, default=64, help='Tile overlap')
    args = parser.parse_args()

    providers = ['CUDAExecutionProvider']
    ort_session = onnxruntime.InferenceSession(args.model_path, providers=providers)
    input_name = ort_session.get_inputs()[0].name

    os.makedirs(args.output, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(args.input, '*')))
    total_images = len(image_paths)
    start_time = time.time()

    with tqdm(total=total_images, desc="Processing images", unit="image") as pbar_images:
        for idx, path in enumerate(image_paths):
            imgname = os.path.splitext(os.path.basename(path))[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = img.astype(np.float32) / 255.

            original_height, original_width = img.shape[:2]
            output_shape = (original_height * args.scale, original_width * args.scale, 3)
            output_img = np.zeros(output_shape, dtype=np.float32)

            tile_size = args.tile
            tile_overlap = args.tile_overlap
            if tile_size is None:
                tile_size = max(original_height, original_width)

            num_tiles_x = (original_width + tile_size - tile_overlap - 1) // (tile_size - tile_overlap)
            num_tiles_y = (original_height + tile_size - tile_overlap - 1) // (tile_size - tile_overlap)

            poisson_result = np.zeros(output_shape, dtype=np.float32)

            with tqdm(total=num_tiles_x * num_tiles_y, desc=f"Processing tiles ({imgname})", unit="tile", leave=False) as pbar_tiles:
                for y in range(num_tiles_y):
                    for x in range(num_tiles_x):
                        x1 = x * (tile_size - tile_overlap)
                        y1 = y * (tile_size - tile_overlap)
                        x2 = min(x1 + tile_size, original_width)
                        y2 = min(y1 + tile_size, original_height)

                        tile_img = img[y1:y2, x1:x2, :]
                        tile_img = pad_image(tile_img, 16)
                        tile_img = tile_img.astype(np.float16)

                        tile_img = np.transpose(tile_img, (2, 0, 1))
                        tile_img = np.expand_dims(tile_img, axis=0)
                        try:
                            output_tile = ort_session.run(None, {input_name: tile_img})[0]
                        except Exception as error:
                            print('Error', error, imgname)
                            continue

                        output_tile = np.clip(output_tile, 0, 1)
                        output_tile = np.transpose(output_tile[0, :, :, :], (1, 2, 0))

                        output_tile = output_tile[:(y2 - y1) * args.scale, :(x2 - x1) * args.scale, :]

                        roi_x1 = x1 * args.scale
                        roi_y1 = y1 * args.scale
                        roi_x2 = min(roi_x1 + (x2 - x1) * args.scale, output_shape[1])
                        roi_y2 = min(roi_y1 + (y2 - y1) * args.scale, output_shape[0])

                        tile_mask = np.zeros((output_tile.shape[0], output_tile.shape[1]), dtype=np.uint8)
                        tile_mask[:] = 255

                        output_tile_uint8 = (output_tile * 255).astype(np.uint8)
                        poisson_result_roi = poisson_result[roi_y1:roi_y2, roi_x1:roi_x2]

                        if output_tile_uint8.shape[:2] == poisson_result_roi.shape[:2]:
                            poisson_result_roi[:] = (poisson_result_roi * (1 - tile_mask[:, :, np.newaxis] / 255) +
                                                      output_tile_uint8 * (tile_mask[:, :, np.newaxis] / 255)).astype(np.uint8)

                        pbar_tiles.update(1)

            output_img = poisson_result

            cv2.imwrite(os.path.join(args.output, f'{imgname}_DRCT-L_X4.png'), output_img)

            pbar_images.update(1)

            elapsed_time = time.time() - start_time
            estimated_remaining_time = (elapsed_time / (idx + 1)) * (total_images - idx - 1)
            pbar_images.set_postfix({
                "Estimated time": f"{estimated_remaining_time:.1f}s"
            })

def pad_image(img, factor):
    height, width = img.shape[:2]
    pad_height = (factor - (height % factor)) % factor
    pad_width = (factor - (width % factor)) % factor
    padded_img = cv2.copyMakeBorder(img, 0, pad_height, 0, pad_width, cv2.BORDER_REFLECT_101)
    return padded_img

if __name__ == '__main__':
    main()