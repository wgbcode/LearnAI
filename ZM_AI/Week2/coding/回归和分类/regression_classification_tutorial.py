"""
监督学习：回归与分类教学演示
包含代码示例和可视化图表
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification, make_regression
import seaborn as sns
import os

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')

def setup_chinese_font():
    """设置中文字体，自动检测系统中可用的中文字体"""
    import platform
    
    # Windows系统字体路径
    windows_font_paths = [
        r'C:\Windows\Fonts\msyh.ttc',      # 微软雅黑
        r'C:\Windows\Fonts\simhei.ttf',    # 黑体
        r'C:\Windows\Fonts\simsun.ttc',    # 宋体
        r'C:\Windows\Fonts\kaiti.ttf',    # 楷体
        r'C:\Windows\Fonts\simfang.ttf',   # 仿宋
    ]
    
    # 获取所有字体名称和路径
    all_fonts_dict = {f.name: f.fname for f in fm.fontManager.ttflist}
    all_fonts = list(all_fonts_dict.keys())
    
    # 常见的中文字体列表（按优先级排序）
    chinese_font_candidates = [
        'Microsoft YaHei',      # 微软雅黑（Windows常见）
        'SimHei',               # 黑体（Windows常见）
        'SimSun',               # 宋体（Windows常见）
        'KaiTi',                # 楷体（Windows常见）
        'FangSong',             # 仿宋（Windows常见）
        'Microsoft JhengHei',   # 微软正黑体
        'STSong',               # 华文宋体
        'STHeiti',              # 华文黑体
        'STKaiti',              # 华文楷体
        'STFangsong',           # 华文仿宋
        'PingFang SC',          # 苹方（macOS）
        'Hiragino Sans GB',     # 冬青黑体（macOS）
        'WenQuanYi Micro Hei',  # 文泉驿微米黑（Linux）
        'Noto Sans CJK SC',     # Noto字体（Linux）
    ]
    
    # 方法1: 如果是Windows系统，先尝试直接使用系统字体路径
    if platform.system() == 'Windows':
        for font_path in windows_font_paths:
            if os.path.exists(font_path):
                try:
                    prop = fm.FontProperties(fname=font_path)
                    plt.rcParams['font.family'] = prop.get_name()
                    plt.rcParams['font.sans-serif'] = [prop.get_name()]
                    print(f"已设置中文字体: {prop.get_name()} (从系统路径: {font_path})")
                    plt.rcParams['axes.unicode_minus'] = False
                    sns.set_style("whitegrid")
                    plt.rcParams['figure.figsize'] = (12, 8)
                    sns.set(font=prop.get_name())
                    return prop.get_name()
                except Exception:
                    continue
    
    # 方法2: 从已注册的字体中查找
    available_font = None
    for font in chinese_font_candidates:
        if font in all_fonts:
            available_font = font
            break
    
    # 如果没找到，尝试查找包含'Microsoft'或'Sim'的字体
    if available_font is None:
        for font_name in all_fonts:
            if 'Microsoft' in font_name or 'Sim' in font_name:
                available_font = font_name
                break
    
    # 如果还是找不到，尝试查找包含中文相关的关键词
    if available_font is None:
        keywords = ['YaHei', 'Hei', 'Song', 'Kai', 'Fang']
        for font_name in all_fonts:
            if any(keyword in font_name for keyword in keywords):
                available_font = font_name
                break
    
    # 设置字体
    if available_font:
        # 尝试直接设置字体文件路径（更可靠）
        try:
            font_path = all_fonts_dict.get(available_font)
            if font_path and os.path.exists(font_path):
                # 使用字体文件路径设置
                prop = fm.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = prop.get_name()
                plt.rcParams['font.sans-serif'] = [prop.get_name()]
                print(f"已设置中文字体: {available_font} (从字体文件: {font_path})")
            else:
                # 使用字体名称设置
                plt.rcParams['font.sans-serif'] = [available_font]
                print(f"已设置中文字体: {available_font}")
        except Exception as e:
            # 如果设置失败，使用字体名称
            plt.rcParams['font.sans-serif'] = [available_font]
            print(f"已设置中文字体: {available_font} (使用字体名称)")
    else:
        # 如果还是找不到，使用默认设置
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    
    # 设置负号正常显示
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置样式
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 确保seaborn也使用中文字体
    if available_font:
        sns.set(font=available_font)
    elif platform.system() == 'Windows':
        # Windows系统默认使用微软雅黑
        sns.set(font='Microsoft YaHei')
    
    # 返回字体名称
    return available_font if available_font else 'Microsoft YaHei'

# 初始化中文字体设置
setup_chinese_font()

# ============================================================================
# 第一部分：回归（Regression）
# ============================================================================

def part1_regression_concept():
    """第一部分：回归概念讲解"""
    print("=" * 80)
    print("第一部分：回归（Regression）")
    print("=" * 80)
    print("\n【概念】")
    print("回归是预测连续数值输出的监督学习任务。")
    print("例如：预测房价、预测温度、预测销售额等。")
    print("\n【特点】")
    print("- 输出是连续数值")
    print("- 目标是找到输入特征与输出之间的函数关系")
    print("- 评估指标：均方误差(MSE)、R²分数等")
    print("\n" + "-" * 80)

def demo_linear_regression():
    """演示：线性回归"""
    print("\n【示例1：线性回归】")
    print("预测房屋面积与价格的关系")
    
    # 生成示例数据
    np.random.seed(42)
    area = np.random.uniform(50, 200, 100).reshape(-1, 1)
    price = 5000 * area.flatten() + np.random.normal(0, 50000, 100)
    
    # 训练模型
    model = LinearRegression()
    model.fit(area, price)
    
    # 预测
    area_pred = np.linspace(50, 200, 100).reshape(-1, 1)
    price_pred = model.predict(area_pred)
    
    # 评估
    price_train_pred = model.predict(area)
    mse = mean_squared_error(price, price_train_pred)
    r2 = r2_score(price, price_train_pred)
    
    print(f"模型方程: 价格 = {model.coef_[0]:.2f} × 面积 + {model.intercept_:.2f}")
    print(f"均方误差(MSE): {mse:.2f}")
    print(f"R²分数: {r2:.4f}")
    
    # 可视化
    plt.figure(figsize=(14, 5))
    
    # 左图：数据点和回归线
    plt.subplot(1, 2, 1)
    plt.scatter(area, price, alpha=0.6, color='blue', label='实际数据')
    plt.plot(area_pred, price_pred, 'r-', linewidth=2, label='回归线')
    plt.xlabel('房屋面积 (平方米)', fontsize=12)
    plt.ylabel('价格 (元)', fontsize=12)
    plt.title('线性回归：房屋面积 vs 价格', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 右图：残差图
    plt.subplot(1, 2, 2)
    residuals = price - price_train_pred
    plt.scatter(price_train_pred, residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('预测值', fontsize=12)
    plt.ylabel('残差', fontsize=12)
    plt.title('残差分析图', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'regression_linear.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存: {output_path}")
    plt.close()

def demo_polynomial_regression():
    """演示：多项式回归"""
    print("\n【示例2：多项式回归】")
    print("处理非线性关系")
    
    # 生成非线性数据
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 2 * X.flatten()**2 - 5 * X.flatten() + 3 + np.random.normal(0, 5, 100)
    
    # 线性回归（对比）
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    y_linear_pred = linear_model.predict(X)
    
    # 多项式回归（2次）
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)
    y_poly_pred = poly_model.predict(X_poly)
    
    # 评估
    mse_linear = mean_squared_error(y, y_linear_pred)
    mse_poly = mean_squared_error(y, y_poly_pred)
    r2_linear = r2_score(y, y_linear_pred)
    r2_poly = r2_score(y, y_poly_pred)
    
    print(f"线性回归 - MSE: {mse_linear:.2f}, R²: {r2_linear:.4f}")
    print(f"多项式回归 - MSE: {mse_poly:.2f}, R²: {r2_poly:.4f}")
    
    # 可视化
    plt.figure(figsize=(14, 5))
    
    # 左图：对比线性回归和多项式回归
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.5, color='blue', label='实际数据', s=30)
    plt.plot(X, y_linear_pred, 'r--', linewidth=2, label='线性回归', alpha=0.7)
    plt.plot(X, y_poly_pred, 'g-', linewidth=2, label='多项式回归(2次)', alpha=0.7)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('线性回归 vs 多项式回归', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 右图：不同次数的多项式回归对比
    plt.subplot(1, 2, 2)
    degrees = [1, 2, 5, 10]
    colors = ['red', 'green', 'blue', 'orange']
    plt.scatter(X, y, alpha=0.3, color='gray', s=20, label='实际数据')
    
    for deg, color in zip(degrees, colors):
        poly_feat = PolynomialFeatures(degree=deg)
        X_poly_deg = poly_feat.fit_transform(X)
        model_deg = LinearRegression()
        model_deg.fit(X_poly_deg, y)
        y_pred_deg = model_deg.predict(X_poly_deg)
        plt.plot(X, y_pred_deg, color=color, linewidth=2, label=f'{deg}次多项式', alpha=0.7)
    
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('不同次数的多项式回归', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'regression_polynomial.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存: {output_path}")
    plt.close()

def demo_regression_comparison():
    """演示：回归问题可视化对比"""
    print("\n【回归问题特点总结】")
    
    # 生成回归数据
    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    
    # 训练模型
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 图1：数据分布
    axes[0].scatter(X, y, alpha=0.6, color='blue')
    axes[0].set_xlabel('特征 X', fontsize=12)
    axes[0].set_ylabel('目标值 y (连续)', fontsize=12)
    axes[0].set_title('回归问题：连续数值输出', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 图2：回归线
    axes[1].scatter(X, y, alpha=0.6, color='blue', label='数据点')
    axes[1].plot(X, y_pred, 'r-', linewidth=2, label='回归线')
    axes[1].set_xlabel('特征 X', fontsize=12)
    axes[1].set_ylabel('目标值 y', fontsize=12)
    axes[1].set_title('拟合回归线', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 图3：预测误差分布
    errors = y - y_pred
    axes[2].hist(errors, bins=20, color='green', alpha=0.7, edgecolor='black')
    axes[2].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[2].set_xlabel('预测误差', fontsize=12)
    axes[2].set_ylabel('频数', fontsize=12)
    axes[2].set_title('预测误差分布', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'regression_overview.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存: {output_path}")
    plt.close()

# ============================================================================
# 第二部分：分类（Classification）
# ============================================================================

def part2_classification_concept():
    """第二部分：分类概念讲解"""
    print("\n\n" + "=" * 80)
    print("第二部分：分类（Classification）")
    print("=" * 80)
    print("\n【概念】")
    print("分类是预测离散类别标签的监督学习任务。")
    print("例如：邮件分类（垃圾/正常）、图像识别（猫/狗）、疾病诊断（患病/健康）等。")
    print("\n【特点】")
    print("- 输出是离散的类别标签")
    print("- 目标是找到决策边界，将不同类别分开")
    print("- 评估指标：准确率、精确率、召回率、F1分数等")
    print("\n" + "-" * 80)

def demo_binary_classification():
    """演示：二分类问题"""
    print("\n【示例1：二分类问题】")
    print("预测学生是否通过考试（通过/不通过）")
    
    # 生成二分类数据
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                               n_informative=2, n_clusters_per_class=1, 
                               random_state=42)
    
    # 训练逻辑回归模型
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    # 预测
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    print(f"模型准确率: {accuracy:.4f}")
    print(f"\n分类报告:")
    print(classification_report(y, y_pred, target_names=['不通过', '通过']))
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：数据点和决策边界
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[0].contourf(xx, yy, Z, alpha=0.4, cmap=cm.RdYlBu)
    scatter = axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap=cm.RdYlBu, 
                             edgecolors='black', s=50)
    axes[0].set_xlabel('特征1 (学习时间)', fontsize=12)
    axes[0].set_ylabel('特征2 (作业完成度)', fontsize=12)
    axes[0].set_title('二分类：决策边界和分类结果', fontsize=14, fontweight='bold')
    axes[0].legend(handles=scatter.legend_elements()[0], 
                   labels=['不通过', '通过'], loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # 右图：混淆矩阵
    cm_matrix = confusion_matrix(y, y_pred)
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['不通过', '通过'],
                yticklabels=['不通过', '通过'])
    axes[1].set_xlabel('预测标签', fontsize=12)
    axes[1].set_ylabel('真实标签', fontsize=12)
    axes[1].set_title('混淆矩阵', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'classification_binary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存: {output_path}")
    plt.close()

def demo_multiclass_classification():
    """演示：多分类问题"""
    print("\n【示例2：多分类问题】")
    print("识别鸢尾花种类（3个类别）")
    
    # 生成多分类数据（3类）
    X, y = make_classification(n_samples=300, n_features=2, n_informative=2,
                               n_redundant=0, n_classes=3, n_clusters_per_class=1,
                               random_state=42)
    
    # 训练逻辑回归模型（多分类）
    model = LogisticRegression(multi_class='multinomial', random_state=42, max_iter=1000)
    model.fit(X, y)
    
    # 预测
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    print(f"模型准确率: {accuracy:.4f}")
    print(f"\n分类报告:")
    print(classification_report(y, y_pred, target_names=['类别0', '类别1', '类别2']))
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：决策区域
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[0].contourf(xx, yy, Z, alpha=0.4, cmap=cm.Set3)
    scatter = axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap=cm.Set3,
                             edgecolors='black', s=50)
    axes[0].set_xlabel('特征1 (花瓣长度)', fontsize=12)
    axes[0].set_ylabel('特征2 (花瓣宽度)', fontsize=12)
    axes[0].set_title('多分类：决策区域', fontsize=14, fontweight='bold')
    axes[0].legend(handles=scatter.legend_elements()[0],
                   labels=['类别0', '类别1', '类别2'], loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # 右图：混淆矩阵
    cm_matrix = confusion_matrix(y, y_pred)
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1],
                xticklabels=['类别0', '类别1', '类别2'],
                yticklabels=['类别0', '类别1', '类别2'])
    axes[1].set_xlabel('预测标签', fontsize=12)
    axes[1].set_ylabel('真实标签', fontsize=12)
    axes[1].set_title('混淆矩阵', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'classification_multiclass.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存: {output_path}")
    plt.close()

def demo_classification_comparison():
    """演示：分类问题可视化对比"""
    print("\n【分类问题特点总结】")
    
    # 生成分类数据
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1,
                               random_state=42)
    
    # 训练模型
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 图1：数据分布（不同类别用不同颜色）
    scatter1 = axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap=cm.RdYlBu, 
                               edgecolors='black', s=50)
    axes[0].set_xlabel('特征1', fontsize=12)
    axes[0].set_ylabel('特征2', fontsize=12)
    axes[0].set_title('分类问题：离散类别输出', fontsize=14, fontweight='bold')
    axes[0].legend(handles=scatter1.legend_elements()[0],
                   labels=['类别0', '类别1'], loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # 图2：决策边界
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[1].contourf(xx, yy, Z, alpha=0.4, cmap=cm.RdYlBu)
    scatter2 = axes[1].scatter(X[:, 0], X[:, 1], c=y, cmap=cm.RdYlBu,
                               edgecolors='black', s=50)
    axes[1].set_xlabel('特征1', fontsize=12)
    axes[1].set_ylabel('特征2', fontsize=12)
    axes[1].set_title('决策边界', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # 图3：预测正确/错误分布
    correct = (y == y_pred)
    colors_map = ['red' if not c else 'green' for c in correct]
    axes[2].scatter(X[:, 0], X[:, 1], c=colors_map, alpha=0.6, s=50,
                   edgecolors='black')
    axes[2].set_xlabel('特征1', fontsize=12)
    axes[2].set_ylabel('特征2', fontsize=12)
    axes[2].set_title('预测结果（绿色=正确，红色=错误）', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'classification_overview.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存: {output_path}")
    plt.close()

# ============================================================================
# 第三部分：回归 vs 分类对比
# ============================================================================

def part3_comparison():
    """第三部分：回归与分类对比"""
    print("\n\n" + "=" * 80)
    print("第三部分：回归 vs 分类 - 核心区别")
    print("=" * 80)
    
    # 创建对比图
    fig = plt.figure(figsize=(16, 10))
    
    # 回归示例
    ax1 = plt.subplot(2, 2, 1)
    X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    model_reg = LinearRegression()
    model_reg.fit(X_reg, y_reg)
    y_reg_pred = model_reg.predict(X_reg)
    ax1.scatter(X_reg, y_reg, alpha=0.6, color='blue', s=30)
    ax1.plot(X_reg, y_reg_pred, 'r-', linewidth=2)
    ax1.set_xlabel('特征 X', fontsize=12)
    ax1.set_ylabel('目标值 y (连续)', fontsize=12)
    ax1.set_title('回归：预测连续数值', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 分类示例
    ax2 = plt.subplot(2, 2, 2)
    X_clf, y_clf = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                       n_informative=2, n_clusters_per_class=1,
                                       random_state=42)
    model_clf = LogisticRegression(random_state=42)
    model_clf.fit(X_clf, y_clf)
    h = 0.02
    x_min, x_max = X_clf[:, 0].min() - 1, X_clf[:, 0].max() + 1
    y_min, y_max = X_clf[:, 1].min() - 1, X_clf[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax2.contourf(xx, yy, Z, alpha=0.4, cmap=cm.RdYlBu)
    scatter = ax2.scatter(X_clf[:, 0], X_clf[:, 1], c=y_clf, cmap=cm.RdYlBu,
                         edgecolors='black', s=50)
    ax2.set_xlabel('特征1', fontsize=12)
    ax2.set_ylabel('特征2', fontsize=12)
    ax2.set_title('分类：预测离散类别', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 输出类型对比
    ax3 = plt.subplot(2, 2, 3)
    ax3.axis('off')
    comparison_text = """
    回归 vs 分类
    
    输出类型：
    • 回归：连续数值（如：价格、温度、年龄）
    • 分类：离散类别（如：是/否、A/B/C、好/中/差）
    
    目标：
    • 回归：找到最佳拟合曲线/平面
    • 分类：找到最佳决策边界
    
    评估指标：
    • 回归：MSE、RMSE、R²、MAE
    • 分类：准确率、精确率、召回率、F1
    
    应用场景：
    • 回归：房价预测、股票预测、销量预测
    • 分类：垃圾邮件检测、图像识别、疾病诊断
    """
    ax3.text(0.1, 0.5, comparison_text, fontsize=13, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 模型选择指南
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    guide_text = """
    如何选择回归还是分类？
    
    1. 看输出是什么？
       → 连续数值 → 回归
       → 类别标签 → 分类
    
    2. 看问题类型？
       → "多少？" → 回归
       → "哪个？" → 分类
    
    3. 看评估方式？
       → 误差大小 → 回归
       → 正确率 → 分类
    
    示例：
    • "预测明天温度" → 回归（温度是连续值）
    • "判断明天是否下雨" → 分类（是/否是离散类别）
    """
    ax4.text(0.1, 0.5, guide_text, fontsize=13, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'regression_vs_classification.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"对比图已保存: {output_path}")
    plt.close()
    
    print("\n【核心区别总结】")
    print("1. 输出类型：回归输出连续值，分类输出离散类别")
    print("2. 目标函数：回归最小化误差，分类最大化分类准确率")
    print("3. 评估指标：回归用MSE/R²，分类用准确率/精确率/召回率")
    print("4. 可视化：回归用曲线/平面，分类用决策边界")

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数：运行完整的教学演示"""
    print("\n" + "=" * 80)
    print("监督学习：回归与分类教学演示")
    print("=" * 80)
    
    # 确保中文字体设置生效（在生成图表前）
    font_name = setup_chinese_font()
    print(f"当前使用字体: {font_name}\n")
    
    # 创建输出文件夹
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"\n已创建输出文件夹: {OUTPUT_DIR}")
    else:
        print(f"\n输出文件夹已存在: {OUTPUT_DIR}")
    
    # 第一部分：回归
    part1_regression_concept()
    demo_linear_regression()
    demo_polynomial_regression()
    demo_regression_comparison()
    
    # 第二部分：分类
    part2_classification_concept()
    demo_binary_classification()
    demo_multiclass_classification()
    demo_classification_comparison()
    
    # 第三部分：对比
    part3_comparison()
    
    print("\n" + "=" * 80)
    print("教学演示完成！所有图表已生成。")
    print("=" * 80)
    print(f"\n生成的图表文件（保存在脚本目录下的 output 文件夹中）：")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("1. regression_linear.png - 线性回归示例")
    print("2. regression_polynomial.png - 多项式回归示例")
    print("3. regression_overview.png - 回归问题总览")
    print("4. classification_binary.png - 二分类示例")
    print("5. classification_multiclass.png - 多分类示例")
    print("6. classification_overview.png - 分类问题总览")
    print("7. regression_vs_classification.png - 回归与分类对比")

if __name__ == "__main__":
    main()

