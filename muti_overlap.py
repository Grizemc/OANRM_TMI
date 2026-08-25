import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

name = ['75% train dataset',"80% train dataset", "85% train dataset", "90% train dataset", "95% train dataset"]
x_values = [75, 80, 85, 90, 95]

data_y = np.array([
    [3.91210302,3.17008035, 4.5119368  , 3.42326067, 4.79950905],
    [3.26401399,2.66484768, 3.72833737, 2.8022914 , 3.88669758],
    [2.74963075,2.26381603, 3.0725461  , 2.32970271, 3.10202932],
    [2.30529945,1.9050259 , 2.40689387,1.89961678 , 2.39766204],
    [1.74013198,1.39374489, 1.59781744, 1.33132977, 1.94814853]
])
data_y = data_y.transpose(1, 0)
# 创建折线图
for i, y_values in enumerate(data_y, start=1):
    plt.plot(x_values, y_values, label=f'{name[i - 1]}', marker='o')

# 添加标题和标签
plt.xlabel('Mean overlapping probability % (test set)', fontsize=14)
plt.ylabel('EPE mm', fontsize=14)

# 添加图例
plt.legend(numpoints=1, loc='upper right', fontsize=9)
#plt.savefig("不同重叠概率下Hamlyn的误差.png", dpi=600)
# 显示图形
plt.show()

# 设置中文显示
#font = FontProperties(fname=r'C:\Users\szm\AppData\Local\Microsoft\Windows\Fonts\HarmonyOS_Sans_SC_Regular.ttf')
# plt.xlabel('Mean overlapping probability %', fontsize=14, fontproperties=font)
#plt.ylabel('EPE mm', fontsize=14, fontproperties=font)