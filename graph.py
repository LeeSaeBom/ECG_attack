import matplotlib.pyplot as plt

# txt 파일에서 데이터 읽기
epochs = []
losses = []
accuracies = []
f1scores = []

with open('./train_val/1d/val_metrics_layer_3.txt', 'r') as file:
    lines = file.readlines()
    header = lines[0].split()  # 첫 번째 줄의 헤더 읽기
    for line in lines[1:]:  # 데이터는 첫 번째 줄 이후에 있으므로 슬라이싱하여 처리
        values = line.split()
        epoch = int(values[0])
        loss = float(values[1])
        accuracy = float(values[2])
        f1score = float(values[3])
        epochs.append(epoch)
        losses.append(loss)
        accuracies.append(accuracy)
        f1scores.append(f1score)

# 그래프 그리기
plt.plot(epochs, accuracies, label='Accuracy')
plt.plot(epochs, f1scores, label='F1 Score')
plt.xlabel(header[0])  # epoch
plt.ylabel('Value')
plt.title('Accuracy and F1 Score per Epoch')
plt.legend()
plt.show()
