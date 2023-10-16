import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 파일 경로 설정
file_path = "./one/1/one_cycle_1.csv"

# CSV 파일 읽기
df = pd.read_csv(file_path)

# 3번째 열의 데이터 추출
data_column = df.iloc[:, 2]

# FGSM 함수 정의
def fgsm_attack(data, epsilon):
    perturbed_data = data + epsilon * np.sign(np.random.randn(len(data)))
    return perturbed_data

# 시각화용 엡실론 값들 설정
epsilon_values = [0, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035,0.04,0.045,0.05,0.1,0.15,0.2,0.25,0.3]

# 오리지널 데이터 추출
original_data = data_column.values

# 각 엡실론 값에 대한 개별 플롯 생성
for epsilon in epsilon_values:
    plt.figure(figsize=(8, 6))
    plt.plot(original_data, label='Original', linewidth=2, color='black')
    perturbed_data = fgsm_attack(data_column, epsilon)
    plt.plot(perturbed_data, label=f'Perturbed (Epsilon = {epsilon:.3f})')
    plt.title(f'FGSM Attack with Epsilon = {epsilon:.3f}')
    plt.xlabel('Data Point Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
