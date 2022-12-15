import os
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import imutils
#Rotation, Translation 함수로 정리
def Rota_trans(img_orig,cols, rows, angle, scale):
        #회전 진행
        M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, scale)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        #새로운 W`H`생성(이미지 크기조정)
        n_w = int((rows * sin) + (cols * cos))
        n_h = int((rows * cos) + (cols * sin))
        #크기 변화에 따른 중심 이동(회전 행렬 조정)
        M[0, 2] += (n_w / 2) - (cols/2)
        M[1, 2] += (n_h / 2) - (rows/2)  
        img_rota = cv2.warpAffine(img_orig, M, (n_w,n_h))
        cv2.imshow('image', img_rota)  
        return img_rota

# Set the working directory to be the current one
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load a reference image as grayscale
img_orig = cv2.imread('fish.png', 0)
rows,cols = img_orig.shape
print(rows,cols)
#윈도우 창 생성
cv2.namedWindow('image')

cv2.imshow('image', img_orig)
angle = 10
scale =1
while True: # 무한 루프
    keycode = cv2.waitKey() # 키보드 입력 반환 값 저장
    if keycode == 32: # 스페이스바 클릭 시 변경 
        cv2.destroyAllWindows()
        img_res = Rota_trans(img_orig, cols, rows, angle, scale)      
        cv2.imshow('image', img_res)        # 이미지 출력
        # print(angle)
        angle += 10 
        if(angle == 360):                   # 350도까지 회전 후 종료
            break
    elif keycode == 27:                     # ESC 누를 시 종료
        break
    
cv2.destroyAllWindows()
