#layer1
import matplotlib.pyplot as plt

# Initialize lists to store data
epsilons = []
accuracy = []
f1_score = []

layer = 1

# 데이터 로드
with open(f'./fgsm/fgsm_results_layer_{layer}.txt', 'r') as file:
    lines = file.readlines()

for line in lines:
    if line.startswith('Epsilon: '):
        parts = line.split(', ')
        epsilons.append(float(parts[0].replace('Epsilon: ', '')))
        accuracy.append(float(parts[1].replace('Accuracy: ', '').replace('%', '')))
        f1_score.append(float(parts[2].replace('F1-Score: ', '')))

# 데이터 Plotting
plt.figure(figsize=(10, 6))
plt.plot(epsilons, accuracy, label='Accuracy', marker='o')
plt.plot(epsilons, f1_score, label='F1-Score', marker='o')

plt.xlabel('Epsilon')
plt.ylabel('Value')
plt.title(f'Epsilon : Accuracy and F1-Score/layer {layer}')
plt.legend()
plt.grid(True)

# X 축 눈금 설정
plt.xticks(epsilons)

# Y 축 눈금 레이블 설정
plt.gca().set_yticklabels(['{:.1f}'.format(x) for x in plt.gca().get_yticks()])

plt.show()
