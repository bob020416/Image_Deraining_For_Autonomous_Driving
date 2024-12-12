import cv2
import os
import argparse
from tqdm import tqdm
import numpy


from utils import chain, analyze
from sols import *
from derain_filter import derain_filter

from multiprocessing import Pool


def run(input_path, output_path):
    # read image
    # tqdm.write(f"Reading image from {input_path}")
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    img = img[:, : w // 2]
    assert img is not None, f"Failed to read image from {input_path}"

    # convert to LUV color space
    # print("Converting image to LUV color space")
    org = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    # org = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # org = cv2.merge([org, org, org])

    img_y = org[..., 0]

    # print("Running the pipeline")
    # steps = chain(img_y, sol1())
    # steps = chain(img_y, sol2())
    # steps = chain(img_y, sol3(img_y))
    # steps = chain(img_y, sol4())
    steps = chain(img_y, sol5())
    steps = chain(img_y, sol6())

    # print(f"Saving intermediate results to output directory")
    ana = analyze(steps)
    ana = np.vstack(ana)
    # cv2.imwrite("output/analyze.jpg", ana)

    for i, I in enumerate(steps):
        I = numpy.ndarray.astype(I, numpy.uint8)
        steps[i] = cv2.merge([I, org[..., 1], org[..., 2]])
        steps[i] = cv2.cvtColor(steps[i], cv2.COLOR_LUV2BGR)
        steps[i] = np.uint8(steps[i])

    # output = np.vstack(steps)
    # cv2.imwrite("output/steps.jpg", output)

    # print(f"Saving stage-1 result to output directory")
    # final = np.vstack([img, steps[-1]])
    # cv2.imwrite("output/stage-1.jpg", steps[-1])

    # print(f"Running stage-2...")
    final_combine = derain_filter(steps[-1], opt=0, iterations=2)
    # cv2.imwrite("output/stage-2.jpg", final_combine)
    final_combine = numpy.uint8(final_combine)
    cv2.imwrite(output_path, final_combine)


def main():
    # python .\combine.py --input data/man-driving-rain.jpg --output results/output_image.jpg
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument(
        "--input",
        default="data/DID-MDN-training/",
        help="Path to the input image (default: data/default_image.jpg)",
    )

    args = parser.parse_args()
    print(args)

    with Pool(4) as p:
        for dir_path, dir_name, file_names in os.walk(args.input):
            # print(f"Processing images in {dir_path}")
            output_dir = args.input.replace("data", "results")
            os.makedirs(output_dir, exist_ok=True)
            for file_name in file_names:
                output_file = os.path.join(output_dir, file_name)
                file_path = os.path.join(dir_path, file_name)
                if not file_name.endswith(".jpg"):
                    print(f"Skipping {file_path}")
                    continue
                print(f"Processing {file_path}")
                p.apply_async(run, args=(file_path, output_file))

        p.close()
        p.join()


from pyinstrument import Profiler

if __name__ == "__main__":
    with Profiler() as p:
        main()
    print(p.output_text(unicode=True, color=True))
