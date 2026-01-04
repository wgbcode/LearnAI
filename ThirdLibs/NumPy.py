
# 学习文档：https://numpy.net/doc/stable/user/quickstart.html

import numpy as np

print("\n" + "=" * 50)
print("===1、基本示例===")
print("=" * 50)
a = np.arange(15).reshape(3, 5)
print(f"生成的数组：\n{a}")
print(f"数组的维度（行和列）：{a.shape}")
print(f"数组的轴数（x轴和y轴）：{a.ndim}")
print(f"数组中元素的类型：{a.dtype.name}")
print(f"数组中元素的总数：{a.size}")
print(f"数组中每个元素的大小(以字节为单位）：{a.itemsize}字节")
print(f"数组实例的类型:{type(a)}")

print("\n" + "=" * 50)
print("===2、创建 numpy 数组===")
print("=" * 50)
print(f"通过列表创建（一维）：\n{np.array([1,2,3])}")
print(f"通过元组创建（一维）：\n{np.array((1,2,3))}")
print(f"通过列表创建（二维）：\n{np.array([[1,2,3],[4,5,6]])}")
print(f"通过元组创建（二维）：\n{np.array([(1,2,3),(4,5,6)])}")
print(f"创建一个全是0的数组（三行四列）：\n{np.zeros((3,4))}")
print(f"创建一个全是1的数组（三行四列）：\n{np.ones((3,4))}")
print(f"创建一个间隔相等的数组（顾头不顾尾）：\n{np.arange(5,30,5)}")
print(f"创建一个间隔相等的数组（顾头和顾尾）：\n{np.linspace(5,30,5)}")

print("\n" + "=" * 50)
print("===3、索引、切片和迭代==")
print("=" * 50)
print(">>（1）一维数组")
a3 = np.arange(10)**3
print(f"创建的数组：\n{a3}")
print(f"索引：{a3[2]}")
print(f"切片：{a3[2:5]}")
a3[:6:2] = 1000  # 索引0-6，步长2，替换值1000
print(f"迭代（会改变原数组）：{a3}")
print(f"迭代（反转）：{a3[::-1]}")

print("\n>>（2）多维数组")
def f(x,y):
    return x*10 + y
b3 = np.fromfunction(f,(5,4),dtype=int)
print(f"通过格式化函数生成的多维数组：\n{b3}")
print(f"通过行索引和列索引获取元素：{b3[2,3]}")
print(f"范围写法1：\n{b3[:3,:3]}")
print(f"范围写法2：\n{b3[1:3,:3]}")
print(f"范围写法3：\n{b3[1:3,:]}")
print(f"范围写法4：\n{b3[1:3,3]}")
print(f"范围写法5：\n{b3[-1]}") # 等价于 [-1,:]
print("对每一行进行迭代")
for row in b3:
    print(f"行:{row}")
print("对每一个元素进行迭代")
for element in b3.flat:
    print(f"元素：{element}")


print("\n" + "=" * 50)
print("===4、改变数组的形状==")
print("=" * 50)
a4 = np.arange(30).reshape(5,6)
print(f"生成的数组：\n{a4}")
print(f"转换成一维数组：\n{a4.ravel()}")
print(f"转换成多维数组：\n{a4.reshape(3,10)}")
print(f"行和列调换：\n{a4.T}")



print("\n" + "=" * 50)
print("===5、将不同的数组堆叠在一起==")
print("=" * 50)
a5 = np.array([[1,2],[3,4]])
b5 = np.array([[5,6],[7,8]])
print(f"原数组a5:\n{a5}")
print(f"原数组b5:\n{b5}")
print(f"行堆叠：\n{np.vstack((a5,b5))}")
print(f"列堆叠：\n{np.hstack((a5,b5))}")

print("\n" + "=" * 50)
print("===6、深拷贝==")
print("=" * 50)
a6 = np.array([[1,2],[3,4]])
b6 = a6.copy()
print(f"深拷贝后，内存地址不一样：{a6 is b6}")








