import torch
from torch.multiprocessing import Pool

# 你的处理函数
def process_tensor(tensor):
    # 你的处理逻辑
    result = tensor * 2
    return result

if __name__ == '__main__':
    # 设置一些示例输入数据
    input_data = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])]

    # 使用多进程池进行并行处理
    with Pool() as pool:
        results = pool.map(process_tensor, input_data)

    # 打印结果
    print("Input data:", input_data)
    print("Results:", results)