import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

name = ['mix train dataset','75% train dataset',"80% train dataset", "85% train dataset", "90% train dataset", "95% train dataset"]
x_values = [75, 80, 85, 90, 95]

data_y = np.array([
    [2.8175544 ,5.83000709,   3.28685687, 3.69713947  , 4.52327145, 3.82793884],
    [2.12355989,3.87814151,   2.33731724 , 2.58844688, 3.07462387 , 2.76674777],
    [1.72446946 ,2.74621377,1.83613507, 2.05181345  , 2.27131389, 2.08391767],
    [1.43517006 ,2.04921996,1.48394449 , 1.61044045,1.6813661  , 1.63376291],
    [1.09588336 ,1.41352358,1.07034027, 1.15654037, 1.1111608 , 1.17490705]
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