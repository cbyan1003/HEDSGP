import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()
x2_values = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

y0_values = [94.4, 95.3, 95.6, 95.85, 95.99, 96.1, 96.45, 96.6, 96.7, 96.75, 96.8]
y0 = [94.4,94.4,94.4,94.4,94.4,94.4,94.4,94.4,94.4,94.4,94.4]

y1_values = [98.0, 98.1, 98.15, 98.16, 98.2, 98.3, 98.55, 98.8, 98.9, 99.05, 99.1]
y1 = [98.0,98.0,98.0,98.0,98.0,98.0,98.0,98.0,98.0,98.0,98.0]

y2_values = [66.4, 67.0, 67.4, 67.9, 68.1, 68.6, 69.1, 69.7, 69.9, 69.9, 70.0]
y2 = [66.4,66.4,66.4,66.4,66.4,66.4,66.4,66.4,66.4,66.4,66.4]
fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=500)


axs[0].plot(x2_values, y0, color='r', label='node')
axs[0].plot(x2_values, y0_values, color='b', label='node')
axs[0].scatter(x2_values, y0_values, color='b',s = 20, marker='s')

axs[0].set_ylim(91, 100)
axs[0].set_ylabel('Object Recall@3(%)')
axs[0].set_xlabel('Proportion(%)')


axs[1].plot(x2_values, y1, color='r', label='node')
axs[1].plot(x2_values, y1_values, color='b', label='node')
axs[1].scatter(x2_values, y1_values, color='b',s = 20, marker='s')

axs[1].set_ylim(97, 100)
axs[1].set_ylabel('Predicate Recall@3(%)')
axs[1].set_xlabel('Proportion(%)')


axs[2].plot(x2_values, y2, color='r', label='node')
axs[2].plot(x2_values, y2_values, color='b', label='node')
axs[2].scatter(x2_values, y2_values, color='b',s = 20, marker='s')

axs[2].set_ylim(60, 80)
axs[2].set_ylabel('Triplet Recall@3(%)')
axs[2].set_xlabel('Proportion(%)')

# 调整布局
plt.tight_layout()
# plt.tight_layout()
# plt.show()
plt.savefig('/home/ycb/3DSSG_old/files/longtail/new_uncer.png')