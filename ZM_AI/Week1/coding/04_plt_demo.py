# 实验四：Matplotlib 数据可视化入门
# 学习目标：掌握基础绘图、子图布局、样式设置和常用图表类型
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

print("=== 1. 基础折线图 ===")
# 1. 基础折线图
x = np.linspace(0, 10, 100)  # 生成0到10之间的100个点
y = np.sin(x)                # 计算sin值
print(f"生成数据点: x范围[{x.min():.2f}, {x.max():.2f}], y范围[{y.min():.2f}, {y.max():.2f}]")

plt.figure(figsize=(8, 6))   # 设置图形大小
plt.plot(x, y, label='sin(x)', linewidth=2)  # 绘制折线图
plt.xlabel('x轴标签')        # x轴标签
plt.ylabel('y轴标签')        # y轴标签
plt.title('基础折线图示例')   # 图表标题
plt.legend()                 # 显示图例
plt.grid(True, alpha=0.3)    # 显示网格（alpha控制透明度）
plt.show()
print("折线图已显示")

print("\n=== 2. 散点图 ===")
# 2. 散点图
x_scatter = np.random.randn(100)
y_scatter = np.random.randn(100)
colors = np.random.rand(100)
sizes = 1000 * np.random.rand(100)
print(f"生成100个随机散点，颜色和大小随机变化")

plt.figure(figsize=(8, 6))
plt.scatter(x_scatter, y_scatter, c=colors, s=sizes, alpha=0.6, cmap='viridis')
plt.colorbar()  # 显示颜色条
plt.xlabel('X值')
plt.ylabel('Y值')
plt.title('散点图示例')
plt.show()
print("散点图已显示")

print("\n=== 3. 柱状图 ===")
# 3. 柱状图
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]
print(f"类别: {categories}")
print(f"数值: {values}")

plt.figure(figsize=(8, 6))
plt.bar(categories, values, color=['red', 'green', 'blue', 'orange', 'purple'])
plt.xlabel('类别')
plt.ylabel('数值')
plt.title('柱状图示例')
plt.show()
print("柱状图已显示")

print("\n=== 4. 子图布局（subplot）===")
# 4. 子图布局（subplot）
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2x2子图布局
print("创建2x2子图布局，包含4种不同类型的图表")

# 子图1：折线图
x1 = np.linspace(0, 2*np.pi, 100)
axes[0, 0].plot(x1, np.sin(x1), 'b-', label='sin')
axes[0, 0].plot(x1, np.cos(x1), 'r--', label='cos')
axes[0, 0].set_title('三角函数')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 子图2：直方图
data_hist = np.random.normal(100, 15, 1000)
axes[0, 1].hist(data_hist, bins=30, color='skyblue', edgecolor='black')
axes[0, 1].set_title('数据分布直方图')
axes[0, 1].set_xlabel('数值')
axes[0, 1].set_ylabel('频数')
print(f"  子图2: 直方图 - 1000个正态分布数据，均值={data_hist.mean():.2f}")

# 子图3：饼图
sizes_pie = [30, 25, 20, 15, 10]
labels_pie = ['A', 'B', 'C', 'D', 'E']
axes[1, 0].pie(sizes_pie, labels=labels_pie, autopct='%1.1f%%', startangle=90)
axes[1, 0].set_title('饼图示例')
print(f"  子图3: 饼图 - 5个类别，占比分别为: {sizes_pie}")

# 子图4：箱线图
data_box = [np.random.normal(0, std, 100) for std in range(1, 5)]
axes[1, 1].boxplot(data_box, labels=['组1', '组2', '组3', '组4'])
axes[1, 1].set_title('箱线图示例')
axes[1, 1].set_ylabel('数值')
print(f"  子图4: 箱线图 - 4组数据，每组100个样本")

plt.tight_layout()  # 自动调整子图间距
plt.show()
print("子图布局已显示")

print("\n=== 5. 样式设置 ===")
# 5. 样式设置
# 使用样式表
try:
    plt.style.use('seaborn-v0_8-darkgrid')  # 或 'ggplot', 'bmh' 等
    print("使用样式: seaborn-v0_8-darkgrid")
except:
    try:
        plt.style.use('seaborn-darkgrid')
        print("使用样式: seaborn-darkgrid")
    except:
        print("使用默认样式")

# 重新设置中文字体（样式表可能会覆盖之前的设置）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

x_style = np.linspace(0, 10, 100)
plt.figure(figsize=(10, 6))
plt.plot(x_style, np.sin(x_style), 'o-', label='sin(x)', markersize=4)
plt.plot(x_style, np.cos(x_style), 's-', label='cos(x)', markersize=4)
plt.xlabel('X轴', fontsize=12)
plt.ylabel('Y轴', fontsize=12)
plt.title('样式化图表', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.show()
print("样式化图表已显示")