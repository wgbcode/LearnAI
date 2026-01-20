# 实验二：NumPy 数值计算实践
# 学习目标：掌握NumPy数组创建、索引切片、向量化运算和统计函数
import numpy as np

print("=== 1. 创建数组 (ndarray) ===")
# 从Python列表创建数组
my_list = [1, 2, 3, 4, 5]
np_array = np.array(my_list)
print(f"从列表创建的数组: {np_array}")

# 创建特殊数组
zeros_matrix = np.zeros((2, 3))  # 2行3列的全零矩阵
ones_matrix = np.ones((3, 2))    # 3行2列的全1矩阵
seq_array = np.arange(0, 10, 2)  # 从0到10(不含)，步长为2
print(f"全零矩阵(2x3):\n{zeros_matrix}")
print(f"序列数组(步长2): {seq_array}")

print("\n=== 2. 数组索引与切片 ===")
data = np.arange(10, 20)
print(f"原始数据: {data}")
print(f"第一个元素: {data[0]}")
print(f"最后三个元素: {data[-3:]}")
print(f"切片(索引1到4): {data[1:5]}")

print("\n=== 3. 向量化运算与广播机制 ===")
a = np.array([10, 20, 30, 40])
b = np.array([1, 2, 3, 4])
print(f"数组a: {a}")
print(f"数组b: {b}")

# 元素级运算（向量化，无需循环）
element_wise_add = a + b
element_wise_multiply = a * b
print(f"a + b (元素级相加): {element_wise_add}")
print(f"a * b (元素级相乘): {element_wise_multiply}")

# 广播机制：标量与数组运算
broadcast_add = a + 100
broadcast_multiply = a * 2
print(f"a + 100 (广播): {broadcast_add}")
print(f"a * 2 (广播): {broadcast_multiply}")

print("\n=== 4. 数组形状操作 ===")
matrix_2d = np.array([[1, 2, 3], [4, 5, 6]])
shape = matrix_2d.shape
reshaped = matrix_2d.reshape(3, 2)
print(f"原始矩阵:\n{matrix_2d}")
print(f"形状: {shape}")
print(f"重塑后(3x2):\n{reshaped}")

print("\n=== 5. 统计函数 ===")
task_matrix = np.random.randint(1, 101, size=(5, 5))
print(f"随机矩阵(5x5):\n{task_matrix}")
max_val = task_matrix.max()
min_val = task_matrix.min()
row_means = task_matrix.mean(axis=1)
col_sums = task_matrix.sum(axis=0)
std_dev = task_matrix.std()
print(f"最大值: {max_val}")
print(f"最小值: {min_val}")
print(f"每行平均值: {row_means}")
print(f"每列总和: {col_sums}")
print(f"标准差: {std_dev:.2f}")
