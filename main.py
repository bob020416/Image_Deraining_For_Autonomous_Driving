import cv2
import os
import argparse

from utils import chain, analyze
from sols import *
from derain_filter import derain_filter


def main():
    # python .\combine.py --input data/man-driving-rain.jpg --output results/output_image.jpg
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument(
        "--input",
        default="data/man-driving-rain.jpg",
        help="Path to the input image (default: data/default_image.jpg)",
    )
    parser.add_argument(
        "--output",
        default="results/man-driving-rain.jpg",
        help="Path to save the output image (default: results/output_image.jpg)",
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # read image
    print(f"Reading image from {args.input}")
    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    assert img is not None, f"Failed to read image from {args.input}"

    # convert to LUV color space
    print("Converting image to LUV color space")
    org = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    # org = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # org = cv2.merge([org, org, org])

    img_y = org[..., 0]

    print("Running the pipeline")
    # steps = chain(img_y, sol1())
    # steps = chain(img_y, sol2())
    steps = chain(img_y, sol3(img_y))
    # steps = chain(img_y, sol4())

    print(f"Saving intermediate results to output directory")
    ana = analyze(steps)
    ana = np.vstack(ana)
    cv2.imwrite("output/analyze.jpg", ana)

    for i, I in enumerate(steps):
        steps[i] = cv2.merge([I, org[..., 1], org[..., 2]])
        steps[i] = cv2.cvtColor(steps[i], cv2.COLOR_LUV2BGR)

    output = np.vstack(steps)
    cv2.imwrite("output/steps.jpg", output)

    print(f"Saving stage-1 result to output directory")
    final = np.vstack([img, steps[-1]])
    cv2.imwrite("output/stage-1.jpg", final)

    print(f"Running stage-2...")
    final_combine = derain_filter(steps[-1], opt=1, iterations=2)
    cv2.imwrite(args.output, final_combine)


from pyinstrument import Profiler

if __name__ == "__main__":
    with Profiler() as p:
        main()
    print(p.output_text(unicode=True, color=True))
