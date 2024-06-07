import cv2
import numpy as np

# 创建两个图层
layer1 = np.zeros((200, 200, 3), dtype=np.uint8)
layer1[:100, :100] = [255, 0, 0]  # 红色

layer2 = np.zeros((200, 200, 3), dtype=np.uint8)
layer2[50:, 50:] = [0, 255, 0]  # 绿色

# 合并图层
combined = layer1 + layer2

# 显示合并后的图像
cv2.imshow('Combined Layers', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
