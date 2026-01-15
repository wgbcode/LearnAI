"""
Week2 机器学习综合练习

本练习涵盖以下内容：
1. 数据处理（数据清洗、特征工程）
2. 分类任务（判断是否为篮球运动员）
3. 回归任务（房价预测）
4. 模型评估（分类和回归的评估指标）

请按照要求完成以下任务。
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 任务1：数据处理 - 数据清洗和特征工程
# ============================================================================

def task1_data_processing():
    """
    任务1：数据处理
    
    给定一个包含"脏"数据的DataFrame，请完成以下操作：
    1. 处理缺失值（用均值填充数值型特征，用众数填充类别型特征）
    2. 处理异常值（年龄应该在18-100之间，房价应该在0-10000之间）
    3. 对类别型特征进行编码（使用LabelEncoder）
    4. 对数值型特征进行标准化（使用StandardScaler）
    
    返回处理后的DataFrame
    """
    # 创建包含"脏"数据的示例数据
    data = {
        'Name': ['Mike', 'Jerry', 'Bryan', 'Patricia', 'Elodie', 'Remy', 'John', 'Marine', 'Julien', 'Fred'],
        'City': ['Miami', 'New York', 'Orlando', 'Miami', 'Phoenix', 'Chicago', 'New York', 'Miami', None, 'Orlando'],
        'Age': [42, 32, 18, 45, 35, 72, 48, 45, 52, 200],  # 包含异常值200
        'Salary': [5000, 6000, None, 5500, 5800, 7000, 6500, 5200, 6800, 3000],
        'HousePrice': [1000, 1300, 700, 1100, 850, 1500, 1200, 1050, 1400, -100]  # 包含异常值-100
    }
    df = pd.DataFrame(data)
    
    print("=" * 60)
    print("任务1：数据处理")
    print("=" * 60)
    print("\n原始数据：")
    print(df)
    print("\n数据信息：")
    print(df.info())
    print("\n缺失值统计：")
    print(df.isnull().sum())
    
    # TODO: 请在此处完成数据处理
    # 提示：
    # 1. 处理缺失值
    # 2. 处理异常值（Age应该在18-100之间，HousePrice应该在0-10000之间）
    # 3. 对City进行编码
    # 4. 对数值型特征进行标准化
    
    # 你的代码写在这里
    df_processed = df.copy()  # 请修改这里
    
    print("\n处理后的数据：")
    print(df_processed)
    print("\n处理后的数据信息：")
    print(df_processed.info())
    
    return df_processed


# ============================================================================
# 任务2：分类任务 - 判断是否为篮球运动员
# ============================================================================

def task2_classification():
    """
    任务2：分类任务
    
    使用处理后的数据训练一个分类模型，判断一个人是否为篮球运动员。
    
    要求：
    1. 准备训练数据（使用任务1处理后的数据）
    2. 分割训练集和测试集（比例8:2）
    3. 训练一个分类模型（可以使用LogisticRegression或DecisionTreeClassifier）
    4. 在测试集上评估模型性能
    5. 输出准确率、混淆矩阵和分类报告
    
    返回训练好的模型和评估结果
    """
    print("\n" + "=" * 60)
    print("任务2：分类任务 - 判断是否为篮球运动员")
    print("=" * 60)
    
    # 创建训练数据（基于演讲稿中的例子）
    train_data = {
        'City': ['Miami', 'New York', 'Orlando', 'Miami', 'Phoenix', 'Chicago', 'New York'],
        'Age': [42, 32, 18, 45, 35, 72, 48],
        'IsBasketballPlayer': [1, 0, 0, 1, 0, 1, 1]  # 1表示是，0表示否
    }
    df_train = pd.DataFrame(train_data)
    
    # 创建测试数据
    test_data = {
        'City': ['Miami', 'Miami', 'Orlando', 'Boston', 'Phoenix'],
        'Age': [45, 52, 20, 34, 90],
        'IsBasketballPlayer': [1, 1, 0, 0, 0]  # 真实标签（用于评估）
    }
    df_test = pd.DataFrame(test_data)
    
    print("\n训练数据：")
    print(df_train)
    print("\n测试数据：")
    print(df_test)
    
    # TODO: 请在此处完成分类任务
    # 提示：
    # 1. 对City进行编码
    # 2. 准备特征X和标签y
    # 3. 分割数据（如果需要）
    # 4. 训练模型
    # 5. 预测并评估
    
    # 你的代码写在这里
    model = None  # 请修改这里
    accuracy = 0  # 请修改这里
    cm = None  # 请修改这里
    
    print("\n模型准确率：", accuracy)
    print("\n混淆矩阵：")
    print(cm)
    print("\n分类报告：")
    # 请添加分类报告的打印
    
    return model, accuracy, cm


# ============================================================================
# 任务3：回归任务 - 房价预测
# ============================================================================

def task3_regression():
    """
    任务3：回归任务
    
    使用处理后的数据训练一个回归模型，预测房价。
    
    要求：
    1. 准备训练数据（基于演讲稿中的房价预测例子）
    2. 分割训练集和测试集（比例8:2）
    3. 训练一个回归模型（可以使用LinearRegression或DecisionTreeRegressor）
    4. 在测试集上评估模型性能
    5. 输出MAE、MSE、R²指标
    
    返回训练好的模型和评估结果
    """
    print("\n" + "=" * 60)
    print("任务3：回归任务 - 房价预测")
    print("=" * 60)
    
    # 创建训练数据（基于演讲稿中的例子）
    train_data = {
        'SchoolDistrict': [8, 9, 6, 9, 3, 7, 8, 5],
        'Orientation': ['南', '西南', '北', '东南', '南', '东', '南', '西'],
        'Area': [100, 120, 60, 80, 95, 110, 105, 75],
        'Price': [1000, 1300, 700, 1100, 850, 1200, 1150, 900]
    }
    df_train = pd.DataFrame(train_data)
    
    # 创建测试数据
    test_data = {
        'SchoolDistrict': [3, 7, 8, 6],
        'Orientation': ['南', '东', '南', '北'],
        'Area': [95, 110, 105, 60],
        'Price': [850, 1200, 1150, 700]  # 真实标签（用于评估）
    }
    df_test = pd.DataFrame(test_data)
    
    print("\n训练数据：")
    print(df_train)
    print("\n测试数据：")
    print(df_test)
    
    # TODO: 请在此处完成回归任务
    # 提示：
    # 1. 对Orientation进行编码
    # 2. 准备特征X和标签y
    # 3. 分割数据（如果需要）
    # 4. 训练模型
    # 5. 预测并评估（计算MAE、MSE、R²）
    
    # 你的代码写在这里
    model = None  # 请修改这里
    mae = 0  # 请修改这里
    mse = 0  # 请修改这里
    r2 = 0  # 请修改这里
    
    print("\n回归模型评估指标：")
    print(f"MAE (平均绝对误差): {mae:.2f}")
    print(f"MSE (均方误差): {mse:.2f}")
    print(f"R² (决定系数): {r2:.4f}")
    
    return model, mae, mse, r2


# ============================================================================
# 任务4：模型评估分析
# ============================================================================

def task4_model_evaluation():
    """
    任务4：模型评估分析
    
    基于任务2和任务3的结果，完成以下分析：
    1. 分析分类模型的性能（准确率、混淆矩阵的含义）
    2. 分析回归模型的性能（MAE、MSE、R²的含义）
    3. 讨论模型的泛化能力
    4. 提出改进建议
    
    返回分析结果（字符串形式）
    """
    print("\n" + "=" * 60)
    print("任务4：模型评估分析")
    print("=" * 60)
    
    # TODO: 请在此处完成模型评估分析
    # 提示：
    # 1. 解释准确率的含义
    # 2. 解释混淆矩阵中TP、TN、FP、FN的含义
    # 3. 解释MAE、MSE、R²的含义
    # 4. 讨论模型的优缺点
    # 5. 提出改进建议
    
    analysis = """
    请在此处填写你的分析：
    
    1. 分类模型性能分析：
    
    
    2. 回归模型性能分析：
    
    
    3. 模型泛化能力讨论：
    
    
    4. 改进建议：
    
    """
    
    print(analysis)
    return analysis


# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数：依次执行所有任务
    """
    print("\n" + "=" * 60)
    print("Week2 机器学习综合练习")
    print("=" * 60)
    
    # 执行任务1：数据处理
    df_processed = task1_data_processing()
    
    # 执行任务2：分类任务
    classification_model, accuracy, cm = task2_classification()
    
    # 执行任务3：回归任务
    regression_model, mae, mse, r2 = task3_regression()
    
    # 执行任务4：模型评估分析
    analysis = task4_model_evaluation()
    
    print("\n" + "=" * 60)
    print("所有任务完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

