import matplotlib.pyplot as plt

# 准备数据
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 33]

# 创建图表和子图
fig, ax = plt.subplots()
values2 = [45, 56, 78, 33, 23]

# 绘制条形图
ax.bar(categories, values,width = 1.0, color='blue', alpha=0.9)
ax.bar(categories, values2,width = 1.0, color='green', alpha=0.3)
# 添加标签和标题
ax.set_xlabel('Categories')
ax.set_ylabel('Values')
ax.set_title('Bar Chart Example')

# 显示图形

plt.show()