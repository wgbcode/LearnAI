"""
模型评估指标教学演示
重点讲解：MSE、RMSE、R²、准确率、精确率、召回率、F1-score、置信度等指标
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.datasets import make_classification, make_regression
import seaborn as sns
import pandas as pd
import os
import platform

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
# 第一部分：回归模型评估指标
# ============================================================================

def demo_regression_metrics():
    """演示回归模型的评估指标：MSE、RMSE、R²"""
    print("=" * 80)
    print("第一部分：回归模型评估指标")
    print("=" * 80)
    
    # 生成回归数据
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 训练模型
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 计算评估指标
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    
    print("\n【回归模型评估指标】")
    print(f"\n训练集指标:")
    print(f"  MSE (均方误差): {mse_train:.2f}")
    print(f"  RMSE (均方根误差): {rmse_train:.2f}")
    print(f"  MAE (平均绝对误差): {mae_train:.2f}")
    print(f"  R² (决定系数): {r2_train:.4f}")
    
    print(f"\n测试集指标:")
    print(f"  MSE (均方误差): {mse_test:.2f}")
    print(f"  RMSE (均方根误差): {rmse_test:.2f}")
    print(f"  MAE (平均绝对误差): {mae_test:.2f}")
    print(f"  R² (决定系数): {r2_test:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 图1：预测 vs 真实值（训练集）
    axes[0, 0].scatter(y_train, y_train_pred, alpha=0.6, color='blue', s=50)
    axes[0, 0].plot([y_train.min(), y_train.max()], 
                    [y_train.min(), y_train.max()], 
                    'r--', linewidth=2, label='完美预测线')
    axes[0, 0].set_xlabel('真实值', fontsize=12)
    axes[0, 0].set_ylabel('预测值', fontsize=12)
    axes[0, 0].set_title(f'训练集：预测 vs 真实值\nR² = {r2_train:.4f}, RMSE = {rmse_train:.2f}', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 图2：预测 vs 真实值（测试集）
    axes[0, 1].scatter(y_test, y_test_pred, alpha=0.6, color='green', s=50)
    axes[0, 1].plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 
                    'r--', linewidth=2, label='完美预测线')
    axes[0, 1].set_xlabel('真实值', fontsize=12)
    axes[0, 1].set_ylabel('预测值', fontsize=12)
    axes[0, 1].set_title(f'测试集：预测 vs 真实值\nR² = {r2_test:.4f}, RMSE = {rmse_test:.2f}', 
                        fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 图3：残差分布
    residuals_train = y_train - y_train_pred
    residuals_test = y_test - y_test_pred
    axes[1, 0].hist(residuals_train, bins=20, alpha=0.7, color='blue', 
                    label=f'训练集 (MSE={mse_train:.2f})', edgecolor='black')
    axes[1, 0].hist(residuals_test, bins=20, alpha=0.7, color='green', 
                    label=f'测试集 (MSE={mse_test:.2f})', edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('残差 (真实值 - 预测值)', fontsize=12)
    axes[1, 0].set_ylabel('频数', fontsize=12)
    axes[1, 0].set_title('残差分布', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 图4：指标对比
    metrics = ['MSE', 'RMSE', 'MAE']
    train_values = [mse_train, rmse_train, mae_train]
    test_values = [mse_test, rmse_test, mae_test]
    x = np.arange(len(metrics))
    width = 0.35
    axes[1, 1].bar(x - width/2, train_values, width, label='训练集', alpha=0.8, color='blue')
    axes[1, 1].bar(x + width/2, test_values, width, label='测试集', alpha=0.8, color='green')
    axes[1, 1].set_xlabel('评估指标', fontsize=12)
    axes[1, 1].set_ylabel('数值', fontsize=12)
    axes[1, 1].set_title('评估指标对比', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'regression_metrics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {output_path}")
    plt.close()
    
    # 指标解释
    print("\n【指标解释】")
    print("1. MSE (均方误差): 预测值与真实值差的平方的平均值，值越小越好")
    print("   - 对异常值敏感（因为平方）")
    print("   - 单位是目标变量的平方")
    print(f"   - 本例: {mse_test:.2f}")
    print("\n2. RMSE (均方根误差): MSE的平方根，值越小越好")
    print("   - 与目标变量单位相同，更易理解")
    print("   - 同样对异常值敏感")
    print(f"   - 本例: {rmse_test:.2f}")
    print("\n3. R² (决定系数): 模型解释数据变异性的比例，范围[0,1]，越接近1越好")
    print("   - R² = 1: 完美预测")
    print("   - R² = 0: 模型与简单平均值一样好")
    print("   - R² < 0: 模型比简单平均值还差")
    print(f"   - 本例: {r2_test:.4f} (模型解释了{r2_test*100:.2f}%的数据变异性)")

# ============================================================================
# 第二部分：分类模型评估指标
# ============================================================================

def demo_classification_metrics():
    """演示分类模型的评估指标：准确率、精确率、召回率、F1-score"""
    print("\n\n" + "=" * 80)
    print("第二部分：分类模型评估指标")
    print("=" * 80)
    
    # 生成分类数据
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1,
                               random_state=42)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 训练模型
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 预测概率（用于置信度分析）
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # 计算评估指标
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_train = precision_score(y_train, y_train_pred)
    precision_test = precision_score(y_test, y_test_pred)
    recall_train = recall_score(y_train, y_train_pred)
    recall_test = recall_score(y_test, y_test_pred)
    f1_train = f1_score(y_train, y_train_pred)
    f1_test = f1_score(y_test, y_test_pred)
    
    print("\n【分类模型评估指标】")
    print(f"\n训练集指标:")
    print(f"  准确率 (Accuracy): {accuracy_train:.4f}")
    print(f"  精确率 (Precision): {precision_train:.4f}")
    print(f"  召回率 (Recall): {recall_train:.4f}")
    print(f"  F1-score: {f1_train:.4f}")
    
    print(f"\n测试集指标:")
    print(f"  准确率 (Accuracy): {accuracy_test:.4f}")
    print(f"  精确率 (Precision): {precision_test:.4f}")
    print(f"  召回率 (Recall): {recall_test:.4f}")
    print(f"  F1-score: {f1_test:.4f}")
    
    # 混淆矩阵
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    
    print(f"\n训练集混淆矩阵:")
    print(cm_train)
    print(f"\n测试集混淆矩阵:")
    print(cm_test)
    
    # 可视化
    fig = plt.figure(figsize=(18, 12))
    
    # 图1：混淆矩阵（训练集）
    ax1 = plt.subplot(3, 3, 1)
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['类别0', '类别1'],
                yticklabels=['类别0', '类别1'])
    ax1.set_xlabel('预测标签', fontsize=12)
    ax1.set_ylabel('真实标签', fontsize=12)
    ax1.set_title('训练集混淆矩阵', fontsize=14, fontweight='bold')
    
    # 图2：混淆矩阵（测试集）
    ax2 = plt.subplot(3, 3, 2)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', ax=ax2,
                xticklabels=['类别0', '类别1'],
                yticklabels=['类别0', '类别1'])
    ax2.set_xlabel('预测标签', fontsize=12)
    ax2.set_ylabel('真实标签', fontsize=12)
    ax2.set_title('测试集混淆矩阵', fontsize=14, fontweight='bold')
    
    # 图3：指标对比
    ax3 = plt.subplot(3, 3, 3)
    metrics = ['准确率', '精确率', '召回率', 'F1-score']
    train_values = [accuracy_train, precision_train, recall_train, f1_train]
    test_values = [accuracy_test, precision_test, recall_test, f1_test]
    x = np.arange(len(metrics))
    width = 0.35
    ax3.bar(x - width/2, train_values, width, label='训练集', alpha=0.8, color='blue')
    ax3.bar(x + width/2, test_values, width, label='测试集', alpha=0.8, color='green')
    ax3.set_xlabel('评估指标', fontsize=12)
    ax3.set_ylabel('分数', fontsize=12)
    ax3.set_title('评估指标对比', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, rotation=45, ha='right')
    ax3.set_ylim([0, 1])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 图4：ROC曲线
    ax4 = plt.subplot(3, 3, 4)
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
    auc_train = auc(fpr_train, tpr_train)
    auc_test = auc(fpr_test, tpr_test)
    ax4.plot(fpr_train, tpr_train, label=f'训练集 (AUC={auc_train:.3f})', linewidth=2)
    ax4.plot(fpr_test, tpr_test, label=f'测试集 (AUC={auc_test:.3f})', linewidth=2)
    ax4.plot([0, 1], [0, 1], 'r--', linewidth=2, label='随机分类器')
    ax4.set_xlabel('假阳性率 (FPR)', fontsize=12)
    ax4.set_ylabel('真阳性率 (TPR/召回率)', fontsize=12)
    ax4.set_title('ROC曲线', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 图5：精确率-召回率曲线
    ax5 = plt.subplot(3, 3, 5)
    precision_curve_train, recall_curve_train, _ = precision_recall_curve(y_train, y_train_proba)
    precision_curve_test, recall_curve_test, _ = precision_recall_curve(y_test, y_test_proba)
    ap_train = average_precision_score(y_train, y_train_proba)
    ap_test = average_precision_score(y_test, y_test_proba)
    ax5.plot(recall_curve_train, precision_curve_train, 
             label=f'训练集 (AP={ap_train:.3f})', linewidth=2)
    ax5.plot(recall_curve_test, precision_curve_test, 
             label=f'测试集 (AP={ap_test:.3f})', linewidth=2)
    ax5.set_xlabel('召回率 (Recall)', fontsize=12)
    ax5.set_ylabel('精确率 (Precision)', fontsize=12)
    ax5.set_title('精确率-召回率曲线', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 图6：预测概率分布（置信度分析）
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(y_test_proba[y_test == 0], bins=20, alpha=0.7, color='red', 
             label='类别0', edgecolor='black')
    ax6.hist(y_test_proba[y_test == 1], bins=20, alpha=0.7, color='blue', 
             label='类别1', edgecolor='black')
    ax6.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='决策阈值')
    ax6.set_xlabel('预测概率 (置信度)', fontsize=12)
    ax6.set_ylabel('频数', fontsize=12)
    ax6.set_title('预测概率分布（置信度）', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 图7：不同阈值下的指标变化
    ax7 = plt.subplot(3, 3, 7)
    thresholds = np.arange(0.1, 1.0, 0.1)
    precisions = []
    recalls = []
    f1_scores = []
    for threshold in thresholds:
        y_pred_thresh = (y_test_proba >= threshold).astype(int)
        precisions.append(precision_score(y_test, y_pred_thresh))
        recalls.append(recall_score(y_test, y_pred_thresh))
        f1_scores.append(f1_score(y_test, y_pred_thresh))
    ax7.plot(thresholds, precisions, 'o-', label='精确率', linewidth=2)
    ax7.plot(thresholds, recalls, 's-', label='召回率', linewidth=2)
    ax7.plot(thresholds, f1_scores, '^-', label='F1-score', linewidth=2)
    ax7.set_xlabel('决策阈值', fontsize=12)
    ax7.set_ylabel('分数', fontsize=12)
    ax7.set_title('不同阈值下的指标变化', fontsize=14, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 图8：指标关系图（精确率 vs 召回率）
    ax8 = plt.subplot(3, 3, 8)
    ax8.scatter(recalls, precisions, c=thresholds, cmap='viridis', s=100, alpha=0.7)
    ax8.set_xlabel('召回率 (Recall)', fontsize=12)
    ax8.set_ylabel('精确率 (Precision)', fontsize=12)
    ax8.set_title('精确率 vs 召回率（不同阈值）', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax8.collections[0], ax=ax8)
    cbar.set_label('阈值', fontsize=10)
    
    # 图9：分类报告可视化
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    report_text = f"""
分类报告总结

准确率: {accuracy_test:.4f}
  - 所有预测中正确的比例

精确率: {precision_test:.4f}
  - 预测为正类中实际为正类的比例
  - 关注：减少假阳性

召回率: {recall_test:.4f}
  - 实际正类中被正确预测的比例
  - 关注：减少假阴性

F1-score: {f1_test:.4f}
  - 精确率和召回率的调和平均
  - 平衡精确率和召回率

AUC: {auc_test:.4f}
  - ROC曲线下面积
  - 衡量分类器整体性能
    """
    ax9.text(0.1, 0.5, report_text, fontsize=11, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'classification_metrics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {output_path}")
    plt.close()
    
    # 指标解释
    print("\n【指标解释】")
    print("1. 准确率 (Accuracy): 所有预测中正确的比例")
    print(f"   - 公式: (TP + TN) / (TP + TN + FP + FN)")
    print(f"   - 本例: {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
    print("\n2. 精确率 (Precision): 预测为正类中实际为正类的比例")
    print(f"   - 公式: TP / (TP + FP)")
    print(f"   - 关注：减少假阳性（误报）")
    print(f"   - 本例: {precision_test:.4f}")
    print("\n3. 召回率 (Recall): 实际正类中被正确预测的比例")
    print(f"   - 公式: TP / (TP + FN)")
    print(f"   - 关注：减少假阴性（漏报）")
    print(f"   - 本例: {recall_test:.4f}")
    print("\n4. F1-score: 精确率和召回率的调和平均")
    print(f"   - 公式: 2 × (Precision × Recall) / (Precision + Recall)")
    print(f"   - 平衡精确率和召回率")
    print(f"   - 本例: {f1_test:.4f}")
    print("\n5. 置信度: 模型对预测结果的把握程度")
    print("   - 预测概率接近0或1：高置信度")
    print("   - 预测概率接近0.5：低置信度")
    print(f"   - 本例平均置信度: {np.mean(np.maximum(y_test_proba, 1-y_test_proba)):.4f}")

# ============================================================================
# 第三部分：指标对比和选择指南
# ============================================================================

def demo_metrics_comparison():
    """演示不同场景下的指标选择"""
    print("\n\n" + "=" * 80)
    print("第三部分：指标选择指南")
    print("=" * 80)
    
    fig = plt.figure(figsize=(16, 10))
    
    # 图1：回归指标对比
    ax1 = plt.subplot(2, 2, 1)
    ax1.axis('off')
    regression_text = """
回归模型评估指标

MSE (均方误差)
  • 公式: Σ(y_true - y_pred)² / n
  • 特点: 对异常值敏感
  • 单位: 目标变量的平方
  • 适用: 一般回归问题

RMSE (均方根误差)
  • 公式: √MSE
  • 特点: 与目标变量单位相同
  • 单位: 与目标变量相同
  • 适用: 需要直观理解的场景

R² (决定系数)
  • 公式: 1 - SS_res/SS_tot
  • 范围: (-∞, 1]
  • 解释: 模型解释数据变异性的比例
  • 适用: 评估模型整体拟合度

MAE (平均绝对误差)
  • 公式: Σ|y_true - y_pred| / n
  • 特点: 对异常值不敏感
  • 单位: 与目标变量相同
  • 适用: 存在异常值的情况
    """
    ax1.text(0.05, 0.5, regression_text, fontsize=11, verticalalignment='center',
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 图2：分类指标对比
    ax2 = plt.subplot(2, 2, 2)
    ax2.axis('off')
    classification_text = """
分类模型评估指标

准确率 (Accuracy)
  • 公式: (TP + TN) / Total
  • 适用: 类别平衡的数据集
  • 局限: 类别不平衡时不准确

精确率 (Precision)
  • 公式: TP / (TP + FP)
  • 关注: 减少假阳性
  • 适用: 误报成本高的场景
  • 例子: 垃圾邮件检测

召回率 (Recall)
  • 公式: TP / (TP + FN)
  • 关注: 减少假阴性
  • 适用: 漏报成本高的场景
  • 例子: 疾病诊断

F1-score
  • 公式: 2PR / (P + R)
  • 特点: 平衡精确率和召回率
  • 适用: 需要综合评估的场景

AUC-ROC
  • 范围: [0, 1]
  • 解释: ROC曲线下面积
  • 适用: 评估分类器整体性能
    """
    ax2.text(0.05, 0.5, classification_text, fontsize=11, verticalalignment='center',
              bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # 图3：指标选择指南
    ax3 = plt.subplot(2, 2, 3)
    ax3.axis('off')
    guide_text = """
如何选择评估指标？

回归问题：
  → 需要直观理解 → RMSE
  → 存在异常值 → MAE
  → 评估拟合度 → R²
  → 一般情况 → MSE

分类问题：
  → 类别平衡 → 准确率
  → 关注误报 → 精确率
  → 关注漏报 → 召回率
  → 需要平衡 → F1-score
  → 整体评估 → AUC-ROC

实际应用：
  • 房价预测 → RMSE, R²
  • 垃圾邮件 → 精确率, F1-score
  • 疾病诊断 → 召回率, F1-score
  • 推荐系统 → 精确率, 召回率
    """
    ax3.text(0.05, 0.5, guide_text, fontsize=11, verticalalignment='center',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 图4：混淆矩阵详解
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    cm_text = """
混淆矩阵详解

           预测
        正类    负类
真实 正类  TP    FN
    负类  FP    TN

TP (True Positive): 真阳性
  • 实际为正类，预测为正类
  • 正确预测的正类

TN (True Negative): 真阴性
  • 实际为负类，预测为负类
  • 正确预测的负类

FP (False Positive): 假阳性
  • 实际为负类，预测为正类
  • 误报（Type I Error）

FN (False Negative): 假阴性
  • 实际为正类，预测为负类
  • 漏报（Type II Error）

从混淆矩阵计算：
  • 准确率 = (TP+TN) / Total
  • 精确率 = TP / (TP+FP)
  • 召回率 = TP / (TP+FN)
  • F1 = 2PR / (P+R)
    """
    ax4.text(0.05, 0.5, cm_text, fontsize=11, verticalalignment='center',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'metrics_guide.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {output_path}")
    plt.close()
    
    print("\n【指标选择建议】")
    print("1. 回归问题优先使用 RMSE 和 R²")
    print("2. 分类问题根据业务需求选择：")
    print("   - 关注误报（如垃圾邮件）→ 精确率")
    print("   - 关注漏报（如疾病诊断）→ 召回率")
    print("   - 需要平衡 → F1-score")
    print("3. 类别不平衡时，不要只看准确率")
    print("4. 结合多个指标综合评估模型性能")

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("模型评估指标教学演示")
    print("=" * 80)
    
    # 创建输出文件夹
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"\n已创建输出文件夹: {OUTPUT_DIR}")
    else:
        print(f"\n输出文件夹已存在: {OUTPUT_DIR}")
    
    # 确保字体设置（在生成所有图表前）
    font_name = setup_chinese_font()
    print(f"当前使用字体: {font_name}\n")
    
    # 测试字体是否正常工作
    try:
        fig_test, ax_test = plt.subplots(figsize=(6, 4))
        ax_test.text(0.5, 0.5, '中文字体测试：回归和分类', 
                     fontsize=16, ha='center', va='center', transform=ax_test.transAxes)
        ax_test.set_title('字体测试', fontsize=14)
        ax_test.axis('off')
        test_path = os.path.join(OUTPUT_DIR, 'font_test.png')
        plt.savefig(test_path, dpi=100, bbox_inches='tight')
        plt.close(fig_test)
        print(f"字体测试图已保存: {test_path}")
    except Exception as e:
        print(f"字体测试失败: {e}")
    
    # 第一部分：回归模型评估
    demo_regression_metrics()
    
    # 第二部分：分类模型评估
    demo_classification_metrics()
    
    # 第三部分：指标对比和选择指南
    demo_metrics_comparison()
    
    print("\n" + "=" * 80)
    print("教学演示完成！所有图表已生成。")
    print("=" * 80)
    print(f"\n生成的图表文件（保存在脚本目录下的 output 文件夹中）：")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("1. regression_metrics.png - 回归模型评估指标")
    print("2. classification_metrics.png - 分类模型评估指标")
    print("3. metrics_guide.png - 指标选择指南")

if __name__ == "__main__":
    main()

