import numpy as np
from PIL import Image
import os

# 폴더 경로 설정 예시 (적절히 수정해야 함)
folder1 = 'C:/school/NeuLF_backup/result/1102/1to6/roty=90_round/imgs'
folder2 = 'C:/school/NeuLF_backup/result/1102/5to10/roty=90_round/imgs'
output_folder = 'C:/school/NeuLF_backup/result/1102/testBlend'

os.mkdir(output_folder)

# 이미지 블렌딩을 위한 for 루프
for i in range(1, 20):  # MATLAB의 1:19는 파이썬에서는 range(1, 20)
    # 이미지 파일 불러오기
    image1 = Image.open(os.path.join(folder1, f'{80+i}.png'))
    image2 = Image.open(os.path.join(folder2, f'{i}.png'))
    
    # 이미지 블렌딩 가중치 설정
    alpha = 1 - i/20
    
    # 이미지 블렌딩
    # 이미지 데이터를 NumPy 배열로 변환 후 블렌딩 수행
    blended_image = alpha * np.array(image1, dtype=float) + (1 - alpha) * np.array(image2, dtype=float)
    # 결과를 정수형 이미지로 변환 후 다시 이미지 형태로 변환
    blended_image = Image.fromarray(np.uint8(blended_image))
    
    # 결과 이미지 저장
    blended_image.save(os.path.join(output_folder, f'b{i+80}.png'))
    print(i)