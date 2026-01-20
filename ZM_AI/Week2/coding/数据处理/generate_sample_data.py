"""
生成示例数据脚本
生成包含重复值、缺失值和异常值的数据集
"""

import pandas as pd
import numpy as np
import os

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_sample_data(n_samples=200, output_file=None):
    """生成示例数据"""
    if output_file is None:
        output_file = os.path.join(SCRIPT_DIR, 'sample_data.csv')
    np.random.seed(42)  # 设置随机种子以便复现
    
    # 生成基础数据
    data = {
        'ID': range(1, n_samples + 1),
        '姓名': [f'用户{i}' for i in range(1, n_samples + 1)],
        '年龄': np.random.randint(18, 65, n_samples),
        '工资': np.random.normal(8000, 2000, n_samples),
        '工作年限': np.random.randint(0, 20, n_samples),
        '部门': np.random.choice(['技术部', '销售部', '市场部', '人事部', '财务部'], n_samples),
        '评分': np.random.uniform(1, 10, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # 1. 添加重复行（约5%的数据）
    n_duplicates = int(n_samples * 0.05)
    duplicate_indices = np.random.choice(df.index, n_duplicates, replace=False)
    duplicate_rows = df.loc[duplicate_indices].copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    # 2. 添加缺失值
    # 年龄缺失约3%
    age_missing = np.random.choice(df.index, int(len(df) * 0.03), replace=False)
    df.loc[age_missing, '年龄'] = np.nan
    
    # 工资缺失约5%
    salary_missing = np.random.choice(df.index, int(len(df) * 0.05), replace=False)
    df.loc[salary_missing, '工资'] = np.nan
    
    # 工作年限缺失约2%
    exp_missing = np.random.choice(df.index, int(len(df) * 0.02), replace=False)
    df.loc[exp_missing, '工作年限'] = np.nan
    
    # 部门缺失约4%
    dept_missing = np.random.choice(df.index, int(len(df) * 0.04), replace=False)
    df.loc[dept_missing, '部门'] = np.nan
    
    # 评分缺失约3%
    score_missing = np.random.choice(df.index, int(len(df) * 0.03), replace=False)
    df.loc[score_missing, '评分'] = np.nan
    
    # 3. 添加异常值
    # 年龄异常值（负数或过大）
    age_outliers = np.random.choice(df.index, int(len(df) * 0.02), replace=False)
    df.loc[age_outliers, '年龄'] = np.random.choice([-5, -10, 150, 200], len(age_outliers))
    
    # 工资异常值（负数或极大值）
    salary_outliers = np.random.choice(df.index, int(len(df) * 0.03), replace=False)
    df.loc[salary_outliers, '工资'] = np.random.choice([-5000, -10000, 500000, 1000000], len(salary_outliers))
    
    # 工作年限异常值
    exp_outliers = np.random.choice(df.index, int(len(df) * 0.02), replace=False)
    df.loc[exp_outliers, '工作年限'] = np.random.choice([-5, 50, 100], len(exp_outliers))
    
    # 评分异常值（超出1-10范围）
    score_outliers = np.random.choice(df.index, int(len(df) * 0.02), replace=False)
    df.loc[score_outliers, '评分'] = np.random.choice([-5, 15, 20], len(score_outliers))
    
    # 打乱数据顺序
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 保存数据
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"示例数据已生成并保存为: {output_file}")
    print(f"数据形状: {df.shape}")
    print(f"\n数据预览:")
    print(df.head(10))
    print(f"\n缺失值统计:")
    print(df.isnull().sum())
    print(f"\n重复行数量: {df.duplicated().sum()}")

if __name__ == "__main__":
    generate_sample_data(n_samples=200)

