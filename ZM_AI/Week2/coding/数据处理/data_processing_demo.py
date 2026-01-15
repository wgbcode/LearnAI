"""
数据处理演示脚本
功能包括：数据去重、处理缺失值、异常值检测和处理
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')  # 忽略警告信息

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_data(file_path=None):
    """加载数据"""
    if file_path is None:
        file_path = os.path.join(SCRIPT_DIR, 'sample_data.csv')
    print("=" * 60)
    print("步骤1: 加载数据")
    print("=" * 60)
    df = pd.read_csv(file_path)
    print(f"数据形状: {df.shape}")
    print(f"\n前5行数据:")
    print(df.head())
    print(f"\n数据基本信息:")
    print(df.info())
    print(f"\n数据统计描述:")
    print(df.describe())
    return df

def remove_duplicates(df):
    """数据去重"""
    print("\n" + "=" * 60)
    print("步骤2: 数据去重")
    print("=" * 60)
    
    # 检查重复行
    duplicate_count = df.duplicated().sum()
    print(f"发现重复行数量: {duplicate_count}")
    
    if duplicate_count > 0:
        print(f"\n重复行示例:")
        print(df[df.duplicated(keep=False)].head(10))
    
    # 去除重复行
    df_cleaned = df.drop_duplicates()
    print(f"\n去重前数据行数: {len(df)}")
    print(f"去重后数据行数: {len(df_cleaned)}")
    print(f"删除重复行数: {len(df) - len(df_cleaned)}")
    
    return df_cleaned

def handle_missing_values(df):
    """处理缺失值"""
    print("\n" + "=" * 60)
    print("步骤3: 处理缺失值")
    print("=" * 60)
    
    # 检查缺失值
    missing_count = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    
    missing_df = pd.DataFrame({
        '缺失数量': missing_count,
        '缺失百分比': missing_percent
    })
    missing_df = missing_df[missing_df['缺失数量'] > 0].sort_values('缺失数量', ascending=False)
    
    print("缺失值统计:")
    print(missing_df)
    
    if len(missing_df) > 0:
        # 可视化缺失值
        plt.figure(figsize=(10, 6))
        missing_df['缺失数量'].plot(kind='bar')
        plt.title('缺失值统计')
        plt.ylabel('缺失数量')
        plt.xticks(rotation=45)
        plt.tight_layout()
        output_path = os.path.join(SCRIPT_DIR, 'missing_values.png')
        plt.savefig(output_path, dpi=150)
        print(f"\n缺失值可视化图表已保存为: {output_path}")
        plt.close()
    
    # 处理缺失值
    df_cleaned = df.copy()
    
    # 对于数值型列，使用中位数填充
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_cleaned[col].isnull().sum() > 0:
            median_value = df_cleaned[col].median()
            df_cleaned[col].fillna(median_value, inplace=True)
            print(f"\n列 '{col}': 使用中位数 {median_value:.2f} 填充缺失值")
    
    # 对于类别型列，使用众数填充
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_cleaned[col].isnull().sum() > 0:
            mode_value = df_cleaned[col].mode()[0] if len(df_cleaned[col].mode()) > 0 else '未知'
            df_cleaned[col].fillna(mode_value, inplace=True)
            print(f"列 '{col}': 使用众数 '{mode_value}' 填充缺失值")
    
    # 如果仍有缺失值，删除该行
    remaining_missing = df_cleaned.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"\n仍有 {remaining_missing} 个缺失值，删除包含缺失值的行")
        df_cleaned = df_cleaned.dropna()
    
    print(f"\n处理缺失值后数据形状: {df_cleaned.shape}")
    print(f"剩余缺失值数量: {df_cleaned.isnull().sum().sum()}")
    
    return df_cleaned

def detect_outliers_iqr(df, column):
    """使用IQR方法检测异常值"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(df, column, threshold=3):
    """使用Z-score方法检测异常值"""
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    outliers = df[z_scores > threshold]
    return outliers

def handle_outliers(df):
    """处理异常值"""
    print("\n" + "=" * 60)
    print("步骤4: 异常值检测和处理")
    print("=" * 60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("没有数值型列，跳过异常值检测")
        return df
    
    df_cleaned = df.copy()
    outlier_summary = []
    
    for col in numeric_cols:
        print(f"\n--- 分析列: {col} ---")
        
        # IQR方法
        outliers_iqr, lower_bound, upper_bound = detect_outliers_iqr(df_cleaned, col)
        outlier_count_iqr = len(outliers_iqr)
        
        # Z-score方法
        outliers_zscore = detect_outliers_zscore(df_cleaned, col)
        outlier_count_zscore = len(outliers_zscore)
        
        print(f"  IQR方法检测到异常值: {outlier_count_iqr} 个")
        print(f"  正常值范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"  Z-score方法检测到异常值: {outlier_count_zscore} 个")
        
        if outlier_count_iqr > 0:
            print(f"  异常值示例:")
            print(outliers_iqr[[col]].head(5))
            
            # 可视化异常值
            plt.figure(figsize=(12, 5))
            
            # 箱线图
            plt.subplot(1, 2, 1)
            plt.boxplot(df_cleaned[col].dropna())
            plt.title(f'{col} 箱线图')
            plt.ylabel('值')
            
            # 直方图
            plt.subplot(1, 2, 2)
            plt.hist(df_cleaned[col].dropna(), bins=30, edgecolor='black')
            plt.axvline(lower_bound, color='r', linestyle='--', label=f'下界: {lower_bound:.2f}')
            plt.axvline(upper_bound, color='r', linestyle='--', label=f'上界: {upper_bound:.2f}')
            plt.title(f'{col} 分布图')
            plt.xlabel('值')
            plt.ylabel('频数')
            plt.legend()
            
            plt.tight_layout()
            output_path = os.path.join(SCRIPT_DIR, f'outliers_{col}.png')
            plt.savefig(output_path, dpi=150)
            print(f"  异常值可视化图表已保存为: {output_path}")
            plt.close()
            
            # 处理异常值：使用边界值替换（Winsorization）
            df_cleaned.loc[df_cleaned[col] < lower_bound, col] = lower_bound
            df_cleaned.loc[df_cleaned[col] > upper_bound, col] = upper_bound
            print(f"  已使用边界值替换异常值")
        
        outlier_summary.append({
            '列名': col,
            'IQR异常值数量': outlier_count_iqr,
            '下界': lower_bound,
            '上界': upper_bound
        })
    
    # 异常值处理总结
    if outlier_summary:
        print("\n异常值处理总结:")
        summary_df = pd.DataFrame(outlier_summary)
        print(summary_df)
    
    return df_cleaned

def save_cleaned_data(df, output_path=None):
    """保存清洗后的数据"""
    if output_path is None:
        output_path = os.path.join(SCRIPT_DIR, 'cleaned_data.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n清洗后的数据已保存为: {output_path}")

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("数据处理演示程序")
    print("=" * 60)
    
    try:
        # 1. 加载数据
        df = load_data()
        
        # 2. 数据去重
        df = remove_duplicates(df)
        
        # 3. 处理缺失值
        df = handle_missing_values(df)
        
        # 4. 处理异常值
        df = handle_outliers(df)
        
        # 5. 保存清洗后的数据
        print("\n" + "=" * 60)
        print("步骤5: 保存清洗后的数据")
        print("=" * 60)
        save_cleaned_data(df)
        
        print("\n" + "=" * 60)
        print("数据处理完成！")
        print("=" * 60)
        print(f"\n最终数据形状: {df.shape}")
        print(f"\n最终数据预览:")
        print(df.head(10))
        
    except FileNotFoundError:
        print("错误: 找不到数据文件 'sample_data.csv'")
        print("请确保数据文件存在于当前目录")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

