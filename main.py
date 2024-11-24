import cv2
import os
import numpy as np

from utils import freq_domain, analyze, chain
from sols import *


def main():
    img = cv2.imread("data/man-driving-rain.jpg", cv2.IMREAD_COLOR)
    img = cv2.resize(img, (0, 0), fx=1, fy=1)
    print(img.shape)

    org = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    # org = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # org = cv2.merge([org, org, org])

    img_y = org[..., 0]

    # steps = chain(img_y, sol1())
    # steps = chain(img_y, sol2())
    steps = chain(img_y, sol3(img_y))
    # steps = chain(img_y, sol4())

    ana = analyze(steps)
    ana = np.vstack(ana)
    cv2.imwrite("output/analyze.jpg", ana)

    for i, I in enumerate(steps):
        steps[i] = cv2.merge([I, org[..., 1], org[..., 2]])
        steps[i] = cv2.cvtColor(steps[i], cv2.COLOR_LUV2BGR)

    output = np.vstack(steps)
    cv2.imwrite("output/steps.jpg", output)

    test = steps[-1]
    cv2.imwrite("output/test.jpg", test)

    final = np.vstack([img, steps[-1]])
    cv2.imwrite("output/final.jpg", final)


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    start = cv2.getTickCount()
    main()
    end = cv2.getTickCount()
    print((end - start) / cv2.getTickFrequency())
