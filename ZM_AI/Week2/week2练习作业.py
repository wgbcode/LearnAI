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

    # 1、处理缺失值
    # 处理数值型缺失值，用中位数替代
    num_col = df_processed.select_dtypes(include=[np.number]).columns
    for col in num_col:
        if df_processed[col].isnull().sum()>0:
            middle_val = df_processed[col].median()
            df_processed[col].fillna(middle_val, inplace=True)
    # 处理类别型缺失值，用众数替代
    obj_col = df_processed.select_dtypes(include=['object']).columns
    for col in obj_col:
        if df_processed[col].isnull().sum()>0:
            mode_val = df_processed[col].mode()[0] if len(df_processed[col].mode()[0])>0 else '未知'
            df_processed[col].fillna(mode_val, inplace=True)
    # 边界条件处理，如果还有缺失值，就整行丢弃
    remaining_num = df_processed.isnull().sum().sum()
    if remaining_num > 0:
        df_processed = df_processed.dropna()

    # 2、处理异常值(IQR方法/Z-score方法）
    def detect_outliers_iqr(df, column):
        """使用IQR方法检测异常值"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers, lower_bound, upper_bound

    # def detect_outliers_zscore(df, column, threshold=3):
    #     """使用Z-score方法检测异常值"""
    #     z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    #     outliers = df[z_scores > threshold]
    #     return outliers

    # 只对数值型特征进行异常值处理
    num_col = df_processed.select_dtypes(include=[np.number]).columns
    for col in num_col:
        outliers, lower_bound, upper_bound = detect_outliers_iqr(df_processed, col)
        print(f"{col}列异常值数量（IQR方法)：{len(outliers)}")
        if len(outliers) > 0:
            df_processed.loc[df_processed[col] < lower_bound, col] = lower_bound
            df_processed.loc[df_processed[col] > upper_bound, col] = upper_bound


    # 3. 对City进行编码
    # 目的：将文本类型的分类数据转换为数字。机器学习算法只能处理数值型数据
    le = LabelEncoder()
    df_processed['City'] = le.fit_transform(df_processed['City'])

    # 4. 对数值型特征进行标准化
    # 目的：将数值型特征进行标准化，使其具有相同的缩放（均值为0，标准差为1的正态分布）
    num_col = df_processed.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df_processed[num_col] = scaler.fit_transform(df_processed[num_col])
    
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

    """
    理解：
    1、对 city 进行编码的意义？
    （1）机器学习只能识别数值，需要将非数字数据进行转换
    （2）转换方法：LabelEncoder 实例的 transform 方法
    （3）确保训练集和测试集相同 city 能转换一致的方法：fit 方法。原理：先学习，建立统一的转换规则（映射表）
    2、什么是特征X和标签Y？
    （1）机器学习的过程：输入特征 -> 机器学习模型 -> 预测标签
    （2）特征是已知信息，标签是结果。已知信息和结果之间存在相关性
    （3）训练集是为了让模型学习这种相关性，测试集是检验模型对这种相关性的学习效果
    3、如何训练模型？
    （1）先通过 LogisticRegression 创建一个模型实例
    （2）训练：通过 fit 方法，将训练集的数据喂给模型
    （3）测试：使用测试集的特征数据 x_test，先通过 predict 方法拿到预测结果，再和标签数据 y_test 进行对比
    """

    # 1、对 city 进行编码
    city_encoder = LabelEncoder()
    all_cities = list(df_train['City']) + list(df_test['City'])
    city_encoder.fit(all_cities)  # 学习所有城市的编码规则
    df_train['City_encoded'] = city_encoder.transform(df_train['City'])
    df_test['City_encoded'] = city_encoder.transform(df_test['City'])

    # 2、准备训练集和测试集的特征X和标签Y（有监督学习）
    x_train = df_train[['City_encoded', 'Age']].values
    y_train = df_train['IsBasketballPlayer'].values
    x_test = df_test[['City_encoded', 'Age']].values
    y_test = df_test['IsBasketballPlayer'].values

    # 3、训练模型（分类模型，如 true or false）
    model = LogisticRegression(random_state=42)
    model.fit(x_train, y_train)

    # 4、预测并评估
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\n模型准确率：", accuracy)
    print("\n混淆矩阵：")
    print(cm)
    print("\n分类报告：")
    print(report)
    
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

    # 1、编码
    orientation_encoder = LabelEncoder()
    all_orientations = list(df_train['Orientation']) + list(df_test['Orientation'])
    orientation_encoder.fit(all_orientations)  # 学习所有城市的编码规则
    df_train['Orientation_encoded'] = orientation_encoder.transform(df_train['Orientation'])
    df_test['Orientation_encoded'] = orientation_encoder.transform(df_test['Orientation'])

    # 2、准备特征X和标签Y
    x_train = df_train[['SchoolDistrict', 'Orientation_encoded','Area']].values
    y_train = df_train['Price'].values
    x_test = df_test[['SchoolDistrict', 'Orientation_encoded','Area']].values
    y_test = df_test['Price'].values

    # 3、训练模型（回归模型）
    model = LogisticRegression(random_state=42)
    model.fit(x_train, y_train)

    # 4、预测并评估
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
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
    
    一. 分类模型性能分析：
    1、理解混淆矩阵（实际值和预测值）
    （1）预测是否准确：T为True，表示准确；F为False，表示不准确
    （2）如果是"2*2"架构，只有正类和负类，分别对应P和N，即Positive和Negative
    （3）预测准确：TP 和 TN
    （4）预测不准确：FP 和 FN
    2、准确率：（TP+TN) / (TP+TN+FP+FN）
    3、性能分析：分类模型测试准确率为 80%，整体还算可以，但测试数据太少
    
    二. 回归模型性能分析：
    1、理解 MAE、MSE、R²
    （1）MAE：平均绝对误差，即 |预测值-实际值|
    （2）MSE：均方误差，即 （预测值-测试值）²
    （3）R²：决定系数，范围在 [0, 1]，越接近 1，拟合效果越好
    2、性能分析：MAE 和 MSE 为 0，R² 为 1，测试效果完美，但测试数据太少
    
    三. 模型泛化能力讨论：
    1、理解欠拟合和过拟合
    （1）欠拟合：训练集的效果不好，测试集的效果也不好
    （2）过拟合：训练集的效果好，但测试集的效果不好，也就是常说的泛化能力比较差
    2、对泛化能力的理解
    （1）泛化能力：指模型对“未见过的数据”进行测试时的预测能力（拟合效果）
    3、提升模型泛化能力的方法
    （1）增加训练集数据
    （2）使用正则化约束
    （3）减少特征数
    （4）调整参数
    （5）降低模型复杂度
    
    四. 改进建议：
    先增加训练集和测试集的数据，目前数据量太少
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

