import cv2
import os
import numpy as np
import argparse

from utils import freq_domain, analyze, chain
from sols import *
from derain import *


def main():
    # python .\combine.py --input data/man-driving-rain.jpg --output results/output_image.jpg
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument('--input', default='data/man-driving-rain.jpg', help='Path to the input image (default: data/default_image.jpg)')
    parser.add_argument('--output', default='results/output_image.jpg', help='Path to save the output image (default: results/output_image.jpg)')

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # read image
    img = cv2.imread(args.input, cv2.IMREAD_COLOR)

    org = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    # org = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # org = cv2.merge([org, org, org])

    img_y = org[..., 0]

    # steps = chain(img_y, sol1())
    # steps = chain(img_y, sol2())
    steps = chain(img_y, sol3(img_y))
    # steps = chain(img_y, sol4())

    # ana = analyze(steps)
    # ana = np.vstack(ana)
    # cv2.imwrite("output/analyze.jpg", ana)

    for i, I in enumerate(steps):
        steps[i] = cv2.merge([I, org[..., 1], org[..., 2]])
        steps[i] = cv2.cvtColor(steps[i], cv2.COLOR_LUV2BGR)

    # output = np.vstack(steps)
    # cv2.imwrite("output/steps.jpg", output)

    # test = steps[-1]
    # cv2.imwrite("output/test.jpg", test)

    # final = np.vstack([img, steps[-1]])
    # cv2.imwrite("output/final.jpg", final)

    final_combine = derain_filter(steps[-1], opt=1, iterations=1)
    cv2.imwrite(args.output, final_combine)


if __name__ == "__main__":
    start = cv2.getTickCount()
    main()
    end = cv2.getTickCount()
    print((end - start) / cv2.getTickFrequency())
