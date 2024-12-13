import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

from metrics import PSNR, SSIM


def run(input_path, output_path):
    # tqdm.write(f"Reading image from {input_path}")
    # tqdm.write(f"input_path: {input_path}, output_path: {output_path}")
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    assert img is not None, f"Failed to read image from {input_path}"
    h, w = img.shape[:2]
    input = img[:, : w // 2]
    grount_truth = img[:, w // 2 :]
    output = cv2.imread(output_path, cv2.IMREAD_COLOR)

    psnr_org = PSNR(grount_truth, input)
    psnr_our = PSNR(grount_truth, output)

    ssim_org = SSIM(grount_truth, input)
    ssim_our = SSIM(grount_truth, output)
    ret = np.array([[psnr_org, psnr_our, ssim_org, ssim_our]])
    return ret


def main():
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument(
        "--data",
        default="results/DID-MDN-training/",
        help="Path to the output image (default: results/DID-MDN-training/)",
    )

    args = parser.parse_args()
    print(args)

    scores = np.array([], dtype=np.float32).reshape(0, 4)
    bar = tqdm(total=12000)
    psnr_org, psnr_our, ssim_org, ssim_our = 0, 0, 0, 0
    print(f"Processing images in {args.data}")
    for dir_path, dir_name, file_names in os.walk(args.data):
        # print(f"Processing images in {dir_path}")
        output_dir = dir_path
        gt_dir = dir_path.replace("results", "data")
        print(f"Output dir: {output_dir}, GT dir: {gt_dir}")
        for file_name in file_names:
            output_file = os.path.join(output_dir, file_name)
            file_path = os.path.join(gt_dir, file_name)
            if not file_name.endswith(".jpg"):
                print(f"Skipping {file_path}")
                continue
            scores = np.append(scores, run(file_path, output_file), axis=0)
            # print(f"Scores: {scores}, shape: {scores.shape}")
            # print(f"Shape: {scores.shape}")
            # print(f"Mean: {np.mean(scores, axis=0)}")
            psnr_org, psnr_our, ssim_org, ssim_our = np.mean(scores, axis=0)
            # print(
            #     f"[Cur] PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, SAM: {sam:.4f}, SCC: {scc:.4f}"
            # )
            # print(scores)
            bar.update(1)
    bar.close()
    print(f"PSNR: {psnr_org:.4f}, {psnr_our:.4f}")
    print(f"SSIM: {ssim_org:.4f}, {ssim_our:.4f}")


from pyinstrument import Profiler

if __name__ == "__main__":
    with Profiler() as p:
        main()
    print(p.output_text(unicode=True, color=True))
