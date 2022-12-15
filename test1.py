import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def Rota_trans(img_orig, cols, rows, angle, scale):
    # 회전 진행
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # 새로운 W`H`생성(이미지 크기조정)
    n_w = int((rows * sin) + (cols * cos))
    n_h = int((rows * cos) + (cols * sin))
    # 크기 변화에 따른 중심 이동(회전 행렬 조정)
    M[0, 2] += (n_w / 2) - (cols / 2)
    M[1, 2] += (n_h / 2) - (rows / 2)
    img_rota = cv2.warpAffine(img_orig, M, (n_w, n_h))
    cv2.imshow('image', img_rota)
    return img_rota

# Set the working directory to be the current one
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Load a reference image
ref1 = cv2.imread('test_1.png')
ref1 = cv2.cvtColor(ref1, cv2.COLOR_BGR2GRAY)
ref2 = cv2.imread('test_2.png')
ref2 = cv2.cvtColor(ref2, cv2.COLOR_BGR2GRAY)
ref3 = cv2.imread('test_3.png')
ref3 = cv2.cvtColor(ref3, cv2.COLOR_BGR2GRAY)

ref_list = [ref1, ref2, ref3]
# Load a template image as grayscale
template = cv2.imread('fish.png', 0)
w, h = template.shape[::-1]



# Load a reference image as grayscale
mask = cv2.imread('fish.png', 0)
rows,cols = template.shape


cv2.namedWindow('image')
# img_scale_start = cv2.resize(template, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
# ret, mask = cv2.threshold(img_scale_start, 0, 255, cv2.THRESH_BINARY) # 마스크 씌우기
# cv2.imshow('image', img_scale_start)
# angle = 200
# scale = 1.5
th, tw = template.shape[:2]
for ref_img in ref_list:
    img_scale_start = cv2.resize(template, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    ret, mask = cv2.threshold(img_scale_start, 0, 255, cv2.THRESH_BINARY)  # 마스크 씌우기
    # cv2.imshow('image', img_scale_start)
    angle = 0
    scale = 0.5
    th, tw = template.shape[:2]
    while True:
        # keycode = cv2.waitKey(delay=0.1)  # 키보드 입력 반환 값 저장
        # if keycode == 32:  # 스페이스바 클릭 시 변경
        cv2.destroyAllWindows()
        img_res = Rota_trans(template, cols, rows, angle, scale)
        ret, mask_res = cv2.threshold(img_res, 0, 255, cv2.THRESH_BINARY)
        cv2.imshow('image', img_res)  # 이미지 출력

        w, h = img_res.shape[::-1]
        # Apply template matching
        res = cv2.matchTemplate(ref_img, img_res, cv2.TM_CCORR_NORMED, mask=mask_res)
        print(scale)
        print(angle)
        # Thresholding
        threshold = 0.985
        loc = np.where(res >= threshold)  # (array(rows), array(cols))
        if len(loc[0]) != 0:
            # if len(loc) != 0:
            # Draw a bounding box
            ref1 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
            img_res = ref1.copy()

            for pt in zip(*loc[::-1]):
                cv2.rectangle(ref1, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

            # Display results
            titles = ['Original', 'Template Matching']
            images = [ref_img, ref1]
            cv2.imshow("success!",ref1)
            cv2.waitKey()
            scale = 0.5
            angle = 0
            break


        angle += 10
        if (scale == 1.5 and angle == 360):  # scale 1.5에서 350까지 회전 후 종료
            break
        if (angle == 360):  # scale +0.5, angle값 초기화
            scale += 0.5
            angle = 0

        # elif keycode == 27:  # ESC 누를 시 종료
        #     break


