# ============================================
# Week1 综合练习作业
# ============================================
# 作业说明：
# 1. 请完成以下所有题目
# 2. 每个题目都有明确的输出要求
# 3. 请确保代码能够正常运行
# 4. 注意代码规范和注释
# ============================================

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

# ============================================
# 题目1：Python基础与字符串格式化（10分）
# ============================================
# 要求：
# 1. 创建一个变量存储你的姓名
# 2. 创建一个变量存储你的年龄
# 3. 创建一个变量存储你最喜欢的编程语言
# 4. 使用f-string格式化输出：我是[姓名]，今年[年龄]岁，最喜欢的编程语言是[语言]
# ============================================

print("=" * 50)
print("题目1：Python基础与字符串格式化")
print("=" * 50)

# 请在下方编写你的代码
# TODO: 完成题目1
name = 'wgb'
age = 20
lang = 'python'
print(f"我是{name},今年{age}岁，最喜欢的编程语言是{lang}")


# ============================================
# 题目2：NumPy数组操作（20分）
# ============================================
# 要求：
# 1. 创建一个包含10个随机整数（范围1-100）的一维数组
# 2. 计算该数组的最大值、最小值和平均值
# 3. 找出数组中大于50的元素
# 4. 将数组重塑为2行5列的二维数组
# 5. 计算二维数组每行的平均值
# ============================================

print("\n" + "=" * 50)
print("题目2：NumPy数组操作")
print("=" * 50)

# 请在下方编写你的代码
# TODO: 完成题目2
a2 = np.random.randint(1, 101,10)
print(f"创建后的一维数组：\n{a2}")
print(f"最大值：{a2.max()}")
print(f"最小值：{a2.min()}")
print(f"平均值：{a2.mean()}")
print(f"大于50的元素：{a2[a2>50]}")
b2 = a2.reshape(2,5)
print(f"转换后的二维数组（2行5列）：\n{b2}")
print(f"二维数组每行的平均值：{np.mean(b2,axis=1)}")


# ============================================
# 题目3：Pandas数据处理（30分）
# ============================================
# 要求：
# 1. 创建一个包含以下信息的DataFrame：
#    - 姓名：['张三', '李四', '王五', '赵六', '钱七']
#    - 年龄：[25, 30, 28, 35, 22]
#    - 城市：['北京', '上海', '广州', '北京', '深圳']
#    - 薪资：[8000, 12000, 10000, 15000, 7000]
# 2. 显示DataFrame的基本信息（前3行）
# 3. 筛选出年龄大于25岁的员工
# 4. 筛选出来自北京且薪资大于10000的员工
# 5. 计算每个城市的平均薪资
# 6. 按薪资降序排列，并显示薪资最高的员工信息
# ============================================

print("\n" + "=" * 50)
print("题目3：Pandas数据处理")
print("=" * 50)

# 请在下方编写你的代码
# TODO: 完成题目3
df = pd.DataFrame({
   "姓名":['张三', '李四', '王五', '赵六', '钱七'],
   "年龄":[25, 30, 28, 35, 22],
   "城市":['北京', '上海', '广州', '北京', '深圳'],
   "薪资":[8000, 12000, 10000, 15000, 7000]
})
print(f"创建后的 DataFrame：\n{df}")
print(f"只显示前3行：\n{df[:3]}")
print(f"年龄大于25岁的员工：\n{df[df['年龄']>25]}")
print(f"薪资大于10000的员工：\n{df[df['薪资']>10000]}")
print(f"计算每个城市的平均工资：\n{df.groupby('城市')['薪资'].mean()}")
df_sort = df.sort_values(by='薪资',ascending=False)
print(f"按薪资降序排列：\n{df_sort}")
print(f"薪资最高的员工信息：\n{df_sort.iloc[0]}")


# ============================================
# 题目4：数据可视化（30分）
# ============================================
# 要求：
# 1. 设置matplotlib支持中文显示（兼容Windows和Mac）
# 2. 创建2x2的子图布局
# 3. 子图1：绘制sin(x)和cos(x)的折线图（x范围0到2π），添加图例和网格
# 4. 子图2：绘制一个包含5个类别的柱状图，类别为['Syntax', 'Java', 'C++', 'JavaScript', 'Go']，数值为[85, 70, 60, 90, 75]
# 5. 子图3：绘制一个饼图，展示5个城市的占比：[30, 25, 20, 15, 10]，标签为['北京', '上海', '广州', '深圳', '杭州']
# 6. 子图4：绘制一个散点图，x和y都是100个随机数（正态分布），添加颜色映射
# 7. 使用tight_layout调整布局，并保存图片为'作业_可视化结果.png'
# ============================================

print("\n" + "=" * 50)
print("题目4：数据可视化")
print("=" * 50)

# 请在下方编写你的代码
# TODO: 完成题目4
# 1、设置matplotlib支持中文显示（兼容Windows和Mac）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Heiti TC']
matplotlib.rcParams['axes.unicode_minus'] = False

# 2、创建2x2的子图布局
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# 3、子图1：绘制折线图
# （1）创建数据
x1 = np.linspace(0, 2 * np.pi, 100)
# （2）绘制正弦函数
ax_left = axs[0, 0]
line_sin, = ax_left.plot(x1, np.sin(x1), color='blue', label='Sin(x)')
ax_left.set_ylabel('Sin(x)')
# （3）绘制余弦函数
ax_right = ax_left.twinx() # 创建双Y轴
line_cos, = ax_right.plot(x1, np.cos(x1), color='red', label='Cos(x)')
ax_right.set_ylabel('Cos(x)')
# （4）添加网格、图例、标题
ax_left.grid(True) # 添加网格
lines = [line_sin, line_cos]
labels = [line.get_label() for line in lines]
ax_left.legend(lines, labels, loc='upper right') # 添加图例
axs[0, 0].set_title('正弦和余弦')

# 4、子图2：绘制柱状图
axs[0, 1].bar(['Syntax', 'Java', 'C++', 'JavaScript', 'Go'], [85, 70, 60, 90, 75])
axs[0, 1].set_title('柱状图')

# 5. 子图3：绘制饼图
axs[1, 0].pie([30, 25, 20, 15, 10], labels=['北京', '上海', '广州', '深圳', '杭州'])
axs[1, 0].set_title('饼图')

# 6. 子图4：绘制散点图
x = np.random.normal(0, 1, 100) # 均值；标准差；数量
y = np.random.normal(0, 1, 100)
colors = np.sqrt(x**2 + y**2)
scatter_plot = axs[1, 1].scatter(x, y, c=colors, cmap='viridis', alpha=0.7, s=50)
axs[1, 1].set_title('散点图')

# 7. 调整布局和保存
plt.tight_layout()
plt.savefig('作业_可视化结果.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================
# 题目5：综合应用（10分）
# ============================================
# 要求：
# 1. 使用NumPy生成100个学生的数学成绩（正态分布，均值75，标准差10）
# 2. 使用Pandas创建一个DataFrame，包含学生ID（1-100）和成绩
# 3. 计算成绩的统计信息：平均分、最高分、最低分、及格率（>=60）
# 4. 使用matplotlib绘制成绩分布直方图（bins=20），并标注平均分线
# 5. 保存图表为'作业_成绩分布.png'
# ============================================

print("\n" + "=" * 50)
print("题目5：综合应用")
print("=" * 50)

# 请在下方编写你的代码
# TODO: 完成题目5
math_scores = np.random.normal(75, 10, 100) # 均值；标准差；数量
print(f"学生成绩：\n{math_scores}")
df_students = pd.DataFrame({
    'student_id':range(1,101),
    'math_scores':math_scores
})
print(f"DataFrame：\n{df_students}")
average_score = df_students['math_scores'].mean()
print(f"平均分：\n{average_score}")
print(f"最高分：\n{df_students['math_scores'].max()}")
print(f"最低分：\n{df_students['math_scores'].min()}")
print(f"及格率：\n{(df_students['math_scores']>=60).mean()*100}%")

# 绘制直方图
plt.figure(figsize=(12, 6))
n, bins, patches = plt.hist(df_students['math_scores'], bins=20,
                           edgecolor='black', alpha=0.7,
                           color='skyblue', label='成绩分布')
plt.title('100名学生数学成绩分布直方图', fontsize=14, fontweight='bold')
plt.xlabel('数学成绩', fontsize=12)
plt.ylabel('学生人数', fontsize=12)
plt.axvline(average_score, color='red', linestyle='--', linewidth=2,
            label=f'平均分: {average_score:.2f}')
plt.text(average_score + 1, max(n) * 0.9, f'平均分: {average_score:.2f}',
         color='red', fontsize=10, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('作业_成绩分布.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 50)
print("作业完成！")
print("=" * 50)

