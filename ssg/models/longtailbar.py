import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns


old_category = ['none', 'standing on', 'attached to', 'hanging on', 'supported by', 'connected to', 'build in', 'part of']
old_value = [97.4, 92.1, 74.3, 56.0, 08.8, 65.2, 87.2, 37.5]
new_value = [96, 90, 78, 61, 27, 67, 87, 54]

# sorted_data1 = sorted(zip(old_category, old_value), key=lambda x: x[1], reverse=True)
# sorted_category1, sorted_value1 = zip(*sorted_data1)


# 设置条形图的宽度
bar_width = 0.35

# 生成位置数组
r1 = np.arange(len(old_category))
r2 = [x + bar_width for x in r1]

# 绘制条形图
plt.figure(figsize=(10, 5), dpi=500)
plt.bar(r1, old_value, width=bar_width, color='blue', alpha=0.5, label='MonoSSG')
plt.bar(r2, new_value, width=bar_width, color='red', alpha=0.5, label='Ours')

# 设置横轴刻度标签
plt.xlabel('Categories', fontweight='bold')
plt.xticks([r + bar_width/2 for r in range(len(old_category))], old_category, rotation=45)
plt.ylabel('Predicate Recall@1', fontweight='bold')
plt.subplots_adjust(bottom=0.3)

# 设置图例
plt.legend()
plt.savefig('/home/ycb/3DSSG_old/files/longtail/new_longtail20.pdf')

# old_category2 = [
#     'armchair', 'backpack', 'bag', 'ball', 'bar', 'basin', 'basket', 'bath cabinet', 'bathtub', 'bed',
#     'bedside table', 'bench', 'bidet', 'bin', 'blanket', 'blinds', 'board', 'book', 'books', 'bookshelf',
#     'bottle', 'box', 'bread', 'bucket', 'cabinet', 'carpet', 'ceiling', 'chair', 'cleanser', 'clock', 'closet',
#     'clothes', 'clothes dryer', 'clutter', 'coffee machine', 'coffee table', 'commode', 'computer desk', 'couch',
#     'couch table', 'counter', 'cup', 'cupboard', 'curtain', 'cushion', 'cutting board', 'decoration', 'desk',
#     'dining chair', 'dining table', 'door', 'doorframe', 'drawer', 'drum', 'drying machine', 'extractor fan',
#     'fireplace', 'floor', 'flower', 'flowers', 'folder', 'food', 'footstool', 'frame', 'fruit plate', 'garbage',
#     'garbage bin', 'grass', 'hand dryer', 'heater', 'item', 'jacket', 'jar', 'kettle', 'kitchen appliance',
#     'kitchen cabinet', 'kitchen counter', 'kitchen hood', 'ladder', 'lamp', 'laptop', 'laundry basket', 'light',
#     'machine', 'magazine rack', 'menu', 'microwave', 'mirror', 'monitor', 'napkins', 'nightstand', 'object',
#     'objects', 'organizer', 'ottoman', 'oven', 'pack', 'pan', 'paper towel', 'papers', 'pc', 'picture',
#     'pile of books', 'pile of papers', 'pillow', 'pipe', 'plant', 'plate', 'player', 'pot', 'printer', 'rack',
#     'radiator', 'recycle bin', 'refrigerator', 'rocking chair', 'scale', 'screen', 'shelf', 'shoe', 'shoe rack',
#     'shoes', 'showcase', 'shower', 'shower curtain', 'shower floor', 'shower wall', 'side table', 'sink',
#     'soap dish', 'socket', 'sofa', 'sofa chair', 'stair', 'stand', 'stool', 'stove', 'stuffed animal', 'suitcase',
#     'table', 'table lamp', 'telephone', 'toaster', 'toilet', 'toilet brush', 'toilet paper',
#     'toilet paper dispenser', 'towel', 'trash can', 'trashcan', 'tube', 'tv', 'tv stand', 'vase', 'wall',
#     'wardrobe', 'washing machine', 'washing powder', 'window', 'windowsill'
# ]

# old_value2 = [
#     0.850, 0.500, 0.143, 0.000, 0.000, 0.000, 0.143, 0.375, 0.750, 0.833,
#     0.000, 0.556, 0.500, 0.000, 0.512, 0.000, 0.400, 0.133, 0.000, 0.143,
#     0.000, 0.229, 0.000, 0.143, 0.169, 0.333, 0.961, 0.620, 0.000, 0.474,
#     0.000, 0.571, 0.000, 0.097, 0.000, 0.286, 0.400, 0.000, 0.135, 0.600,
#     0.231, 0.000, 0.125, 0.820, 0.197, 0.000, 0.103, 0.421, 0.000, 0.000,
#     0.814, 0.393, 0.000, 0.000, 0.000, 0.000, 0.167, 0.981, 0.000, 0.375,
#     0.000, 0.000, 0.000, 0.143, 0.000, 0.250, 0.000, 0.000, 0.667, 0.486,
#     0.338, 0.000, 0.143, 0.625, 0.000, 0.446, 0.556, 0.167, 0.500, 0.485,
#     0.333, 0.000, 0.409, 1.000, 0.000, 0.000, 0.500, 0.143, 0.938, 0.000,
#     0.000, 0.046, 0.000, 0.000, 0.467, 0.625, 0.000, 0.500, 0.000, 0.000,
#     1.000, 0.682, 0.250, 0.000, 0.740, 0.250, 0.689, 0.100, 0.000, 0.000,
#     0.000, 0.000, 0.625, 0.000, 0.200, 0.500, 0.000, 0.000, 0.518, 0.000,
#     0.000, 0.600, 0.636, 1.000, 1.000, 0.500, 0.000, 0.087, 0.617, 0.000,
#     0.000, 0.722, 0.000, 0.000, 0.154, 0.238, 0.765, 0.000, 0.167, 0.470,
#     0.000, 0.500, 0.000, 1.000, 0.273, 0.667, 0.000, 0.742, 0.562, 0.000,
#     0.000, 0.842, 0.097, 0.135, 0.766, 0.462, 0.571, 0.000, 0.512, 0.529,
# ]

old_category2 = ['attached to', 'behind', 'belonging to', 'bigger than', 'build in', 
                 'close by', 'connected to', 'cover', 'front', 'hanging in', 
                 'hanging on', 'higher than', 'inside', 'leaning against', 'left', 
                 'lower than', 'lying in', 'lying on', 'part of', 'right', 
                 'same as', 'same symmetry as', 'smaller than', 'standing in', 'standing on', 
                 'supported by']

old_value2 = [0.185, 0.550, 0.335, 0.050, 0.456,
              0.68, 0.543, 0.037, 0.630, 1.000, 
              0.510, 0.032, 0.000, 0.029, 0.600,
              0.078, 0.073, 0.717, 0.07, 0.582, 
              0.610, 0.052, 0.199, 0.256, 0.804, 
              0.118]
old_value2 = [x * 100 for x in old_value2]

setorder=['left',
'right',
'close by',
'front',
'behind',
'attached to',
'standing on',
'higher than',
'lower than',
'same as',
'bigger than',
'smaller than',
'lying on',
'hanging on',
'supported by',
'same symmetry as',
'standing in',
'build in',
'leaning against',
'connected to',
'belonging to',
'lying in',
'part of',
'cover',
'hanging in',
'inside']
setvalue=[-1] * 26
new_value3 = [-1] * 26
ourvalue = [21, 48, 35, 10, 48, 
            85.0, 45.6, 7.4, 58, 100, 
            44, 11.7, 0, 9.8, 67.3, 
            10, 10, 72, 24, 60, 
            74, 10, 43, 21, 85, 
            24]

setorder2 = [category for category in old_category2 if category in setorder]

for category in old_category2:
    position = setorder.index(category)
    old_pos = old_category2.index(category)
    setvalue[position] = old_value2[old_pos]
    new_value3[position] = ourvalue[old_pos]
print(setorder)
print(setvalue)
bar_width = 0.35


r3 = np.arange(len(setorder))
r4 = [x + bar_width for x in r3]

plt.figure(figsize=(10,5), dpi=500)
plt.bar(r3, setvalue, width=bar_width, color='blue', alpha=0.5, label='MonoSSG')
plt.bar(r4, new_value3, width=bar_width, color='red', alpha=0.5, label='Ours')
plt.xlabel('Categories', fontweight='bold')
plt.xticks([r + bar_width/2 for r in range(len(setorder))], setorder, rotation=45, ha='right')
plt.ylabel('Predicate Recall@50', fontweight='bold')
plt.subplots_adjust(bottom=0.4)

plt.legend()
plt.savefig('/home/ycb/3DSSG_old/files/longtail/new_longtail160.pdf')