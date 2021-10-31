from matplotlib import pylab as plt
import cv2
import numpy as np


# function to mask everything other than the region of interest
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)  # takes the image matrix as parameter
    # channel_count = img.shape[2]
    match_mask_color = 255  # match colour with the same colour channel counts
    cv2.fillPoly(mask, vertices, match_mask_color)  # fill inside the outer region of intrest part
    masked_image = cv2.bitwise_and(img, mask)  # return the image only where the masked colour matches
    return masked_image


def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)  # merge the images assigning weights
    return img


# image = cv2.imread('road.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
def process(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]

    region_of_interest_vertices = [
        (0, height),
        (width / 2, height * 4/7),
        (width, height)
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 200)  # canny edge detection
    cropped_image = region_of_interest(canny_image,
                                       np.array([region_of_interest_vertices], np.int32, ))

    lines = cv2.HoughLinesP(cropped_image,
                            rho=6,
                            theta=np.pi / 180,
                            threshold=160,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=25)

    image_with_lines = draw_the_lines(image, lines)
    return image_with_lines


cap = cv2.VideoCapture('challenge.mp4')

while(cap.isOpened()):

    ret, frame = cap.read()
    frame = process(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(30)   &   0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()