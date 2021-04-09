from cv2 import cvtColor, COLOR_BGR2GRAY, dilate, erode, filter2D, blur, getRotationMatrix2D, warpAffine, INTER_CUBIC, \
    BORDER_REPLICATE, threshold, THRESH_BINARY_INV, THRESH_OTSU
from os import listdir, path, rename
from numpy import sum, array, arange, ones, asarray, uint8
import pytesseract
from PIL import Image
from re import findall
from scipy.ndimage import interpolation as inter
from pdf2image import convert_from_path

# namedWindow("", WINDOW_NORMAL)
pytesseract.pytesseract.tesseract_cmd = r"lib\Tesseract-OCR\tesseract.exe"
poppler_path = r"lib\poppler\bin"


def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = sum(data, axis=1)
        score = sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cvtColor(image, COLOR_BGR2GRAY)
    thresh = threshold(gray, 0, 255, THRESH_BINARY_INV + THRESH_OTSU)[1]

    scores = []
    angles = arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = getRotationMatrix2D(center, best_angle, 1.0)
    rotated = warpAffine(image, M, (w, h), flags=INTER_CUBIC, \
                         borderMode=BORDER_REPLICATE)

    return best_angle, rotated


def get_string(img, operation=None):
    img = correct_skew(img)[1]
    img = cvtColor(img, COLOR_BGR2GRAY)
    kernel = ones((1, 1), uint8)
    img = dilate(img, kernel, iterations=1)
    img = erode(img, kernel, iterations=1)
    if operation == "sharp":
        kernel = array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = filter2D(img, -1, kernel)
    if operation == "blur":
        img = blur(img, (3, 3))
    result = pytesseract.image_to_string(Image.fromarray(img))

    return result


def fetch_name_from_string(string):
    return findall(r'P0[0-9]{4,10}', string)


input_path = str(input("Enter directory Path : "))
operations = [None, "sharp", "blur"]
for file in listdir(input_path):
    print("Processing", file)
    if file.split(".")[-1] not in ["pdf", "PDF"]:
        continue
    if len(fetch_name_from_string(file.split(".")[0])):
        continue
    images = convert_from_path(path.join(input_path, file), poppler_path=poppler_path)
    for img in images:
        st = False
        for operation in operations:
            name = get_string(asarray(img), operation=operation)
            name = fetch_name_from_string(name)
            if len(name):
                if name[0] + ".pdf" in listdir(input_path):
                    inc = 0
                    while name[0] + "_" + str(inc) + ".pdf" in listdir(input_path):
                        inc += 1
                    rename(path.join(input_path, file), path.join(input_path, name[0] + "_" + str(inc) + ".pdf"))
                else:
                    rename(path.join(input_path, file), path.join(input_path, name[0] + ".pdf"))

                st = True
                break
        if st:
            break

print("All files are successfully renamed.")
