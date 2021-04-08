import cv2
import os
import numpy as np
import pytesseract
from PIL import Image
import re
from scipy.ndimage import interpolation as inter
from pdf2image import convert_from_path

# cv2.namedWindow("", cv2.WINDOW_NORMAL)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
                             borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated


def get_string(img, operation=None):
    img = correct_skew(img)[1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    if operation == "sharp":
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
    if operation == "blur":
        img = cv2.blur(img, (3, 3))
    result = pytesseract.image_to_string(Image.fromarray(img))

    return result


def fetch_name_from_string(string):
    return re.findall(r'P0[0-9]{4,10}', string)


input_path = str(input("Enter directory Path : "))
operations = [None, "sharp", "blur"]
for file in os.listdir(input_path):
    print("Processing", file)
    if file.split(".")[-1] not in ["pdf", "PDF"]:
        continue
    if len(fetch_name_from_string(file.split(".")[0])):
        continue
    images = convert_from_path(os.path.join(input_path, file), poppler_path=r"C:\Program Files\poppler-0.68.0\bin")
    for img in images:
        st = False
        for operation in operations:
            name = get_string(np.asarray(img), operation=operation)
            name = fetch_name_from_string(name)
            if len(name):
                if name[0]+".pdf" in os.listdir(input_path):
                    inc = 0
                    while name[0]+"_"+str(inc)+".pdf" in os.listdir(input_path):
                        inc += 1
                    os.rename(os.path.join(input_path, file), os.path.join(input_path, name[0]+"_"+str(inc)+".pdf"))
                else:
                    os.rename(os.path.join(input_path, file), os.path.join(input_path, name[0] + ".pdf"))
                st = True
                break
        if st:
            break

print("All files are successfully renamed.")

