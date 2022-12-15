import os
import cv2
import numpy as np

# Set the working directory to be the current one
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#이미지 불러오기
img_orig = cv2.imread('fish.png', 0)
rows,cols = img_orig.shape

#윈도우 창 생성
# cv2.namedWindow('image')

#결과
cv2.imshow('image', img_orig)
angle = 10      
while True: # 무한 루프 ㅑ
    keycode = cv2.waitKey() # 키보드 입력 반환 값 저장
    if keycode == 32: # 스페이스바 클릭 시 Rotation
        cv2.destroyAllWindows()  
        M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
        img_res = cv2.warpAffine(img_orig, M, (cols,rows))
        print(angle)        
        angle += 10 
        cv2.imshow('image', img_res) # 이미지 출력
        if(angle == 360): # 350도까지 회전 후 종료
            break   
    elif keycode == 27: # ESC 누를 시 종료
        break            

cv2.destroyAllWindows()