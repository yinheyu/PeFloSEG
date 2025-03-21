import numpy as np
import matplotlib.pyplot as plt

# 定义预测误差
pred_error = np.linspace(0, 1, 500)

# 计算Focal Loss
gamma = 2.0  # Focal Loss中的gamma参数
alpha = 0.25  # Focal Loss中的alpha参数
focal_loss = -alpha * (1 - pred_error)**gamma * np.log(pred_error + 1e-6)

# 计算IoU Loss    
iou_loss = -np.log(1 - pred_error + 1e-6)

# 绘制损失函数曲线
plt.figure(figsize=(10, 6))
plt.plot(pred_error, focal_loss, label='Focal Loss')
plt.plot(pred_error, iou_loss, label='IoU Loss')
plt.xlabel('Prediction Error')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
