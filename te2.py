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
reference= cv2.imread('test_1.png', 0)


# reference1 = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)               

# Load a template image as grayscale
template = cv2.imread('fish.png', 0)
w, h = template.shape[::-1]
# 마스크 생성
mask = cv2.imread('fish.png', 0)
rows,cols = template.shape
#print(rows,cols)
#윈도우 창 생성
cv2.namedWindow('imgae')
img_scale_start = cv2.resize(template, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
ret, mask = cv2.threshold(img_scale_start, 0, 255, cv2.THRESH_BINARY) # 마스크 씌우기
angle = 10
scale = 0.5
th, tw = template.shape[:2]
success = True
while (success == True):  # 무한 루프
    if (success == True):  # 스페이스바 클릭 시 변경
        # cv2.destroyAllWindows()
        img_res = Rota_trans(template, cols, rows, angle, scale)
        ret, mask_res = cv2.threshold(img_res, 0, 255, cv2.THRESH_BINARY)
        # cv2.imshow('image', img_res)  # 이미지 출력 필요없고            
        print(angle)                 #각도 확인용
        w, h = img_res.shape[::-1]     # 박스 그릴때 변경된 이미지의 사이즈로 그리기 위해서 저장
        
        # Apply template matching
        res = cv2.matchTemplate(reference, img_res, cv2.TM_CCORR_NORMED, mask=mask_res)

        # 임계값 설정
        threshold = 0.985
        loc = np.where(res >= threshold)  # (array(rows), array(cols))

        # 결과값 출력
        # cv2.imshow('image', img_res2)
        angle += 10
        if (scale == 1.5 and angle == 360):  # scale 1.5에서 350까지 회전 후 종료
            break
        if (angle == 360):  # scale +0.5, angle값 초기화
            scale += 0.5
            angle = 0
        
        # 박스 그리기
        img_res2 = reference.copy()
        img_res2 = cv2.cvtColor(img_res2, cv2.COLOR_GRAY2BGR)
        # 이미지 찾을시 빨간 BOX 생성
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_res2, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
            # 결과값 출력
            success = False
            break
    if(success == False):
        break
cv2.imshow('image', img_res2)
cv2.waitKey()
cv2.destroyAllWindows()