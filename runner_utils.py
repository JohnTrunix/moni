import random
import cv2
import pkg_resources


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    # plot dot in lower center to indicate ground position
    cv2.circle(img, (int((x[0] + x[2]) / 2), int(x[3])),
               8, color, thickness=1, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def warpPoint(p, H):
    px = (H[0][0] * p[0] + H[0][1] * p[1] + H[0][2]) / \
        ((H[2][0] * p[0] + H[2][1] * p[1] + H[2][2]))
    py = (H[1][0] * p[0] + H[1][1] * p[1] + H[1][2]) / \
        ((H[2][0] * p[0] + H[2][1] * p[1] + H[2][2]))
    return px, py


def check_packages(requirements_path):
    with open(requirements_path, 'r') as f:
        required_packages = f.readlines()

    missing_packages = []

    for package in required_packages:
        package = package.strip()
        try:
            pkg_resources.require(package)
        except pkg_resources.DistributionNotFound:
            missing_packages.append(package)

    if missing_packages:
        print(
            f"ERROR: The following packages are missing: {', '.join(missing_packages)}")
    else:
        print("All required packages are present in the runtime environment.")
