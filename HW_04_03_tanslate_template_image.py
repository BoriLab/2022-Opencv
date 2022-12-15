import os
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

# Set the working directory to be the current one
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load a reference image as grayscale
img_orig = cv2.imread('fish.png', 0)
rows,cols = img_orig.shape
print(rows,cols)
#윈도우 창 생성
cv2.namedWindow('image')
# Translation

cv2.imshow('image', img_orig)
angle = 10
while True: # 무한 루프
    keycode = cv2.waitKey() # 키보드 입력 반환 값 저장
    if keycode == 32: # 스페이스바 클릭 시 변경 
        cv2.destroyAllWindows() 
        #새로운 W`H`생성
        n_w = int(abs(cols * math.cos(angle * (math.pi / 180))) + abs(rows * math.sin(angle * (math.pi / 180))))
        n_h= int(abs(cols * math.sin(angle * (math.pi / 180))) + abs(rows * math.cos(angle * (math.pi / 180))))
        x_t = n_w/2 - cols/2; y_t = n_h/2 - rows/2      #중심이동 거리 구하기 
        # print(n_h,n_w) 
        # print("각도",angle)
        M = np.float32([[1,0,x_t],[0,1,y_t]])           # X, Y 중심이동
        img_res = cv2.warpAffine(img_orig, M, (n_w,n_h))
        angle += 10
        cv2.imshow('image', img_res)        # 이미지 출력
        if(angle == 360):                   # 350도까지 회전 후 종료
            break   
    elif keycode == 27:                     # ESC 누를 시 종료
        break

cv2.destroyAllWindows()