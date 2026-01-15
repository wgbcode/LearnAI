# 实验三：Pandas 数据分析入门
# 学习目标：掌握DataFrame创建、数据读取、选择筛选和统计分析
import pandas as pd
import os

print("=== 1. 创建 DataFrame ===")
# 从字典创建（最常用方式）
data_dict = {
    '姓名': ['小明', '小红', '小刚', '小丽'],
    '年龄': [18, 19, 17, 18],
    '城市': ['北京', '上海', '广州', '北京'],
    '分数': [95, 88, 92, 98]
}
df = pd.DataFrame(data_dict)
print("创建的DataFrame:")
print(df)

print("\n=== 2. 读取外部数据 ===")
try:
    # 读取CSV文件（使用脚本所在目录，确保无论从哪里运行都能找到文件）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'students.csv')
    csv_df = pd.read_csv(csv_path)
    print("成功读取CSV文件")
    
    # 3. 数据查看与基本信息
    print("\n=== 3. 数据查看与基本信息 ===")
    print("数据前3行:")
    print(csv_df.head(3))
    print("\n数据基本信息:")
    csv_df.info()
    print("\n数值数据统计描述:")
    print(csv_df.describe())
    
    # 4. 数据选择与筛选
    print("\n=== 4. 数据选择与筛选 ===")
    # 选择单列（返回Series）
    names = csv_df['name']
    print("所有姓名:")
    print(names)
    
    # 选择多列（返回DataFrame）
    name_and_score = csv_df[['name', 'score']]
    print("\n姓名和分数:")
    print(name_and_score)
    
    # 条件筛选：布尔索引
    high_scores = csv_df[csv_df['score'] > 90]
    print("\n分数高于90分的学生:")
    print(high_scores)
    
    beijing_students = csv_df[csv_df['city'] == '北京']
    print("\n所有来自北京的学生:")
    print(beijing_students)
    
    # 多条件筛选（使用 & 或 |）
    young_high_scores = csv_df[(csv_df['age'] < 23) & (csv_df['score'] > 90)]
    print("\n年龄<23且分数>90的学生:")
    print(young_high_scores)
    
    # 5. 数据统计与聚合
    print("\n=== 5. 数据统计与聚合 ===")
    avg_age = csv_df['age'].mean()
    avg_score = csv_df['score'].mean()
    max_score = csv_df['score'].max()
    print(f"平均年龄: {avg_age:.2f}")
    print(f"平均分数: {avg_score:.2f}")
    print(f"最高分: {max_score}")
    
    # 按列分组统计
    city_avg_score = csv_df.groupby('city')['score'].mean()
    print("\n各城市平均分:")
    print(city_avg_score)
    
    # 6. 数据排序
    print("\n=== 6. 数据排序 ===")
    sorted_by_score = csv_df.sort_values(by='score', ascending=False)
    print("按分数降序排列:")
    print(sorted_by_score)
    top_student = sorted_by_score.iloc[0]
    print(f"\n分数最高的学生: {top_student['name']}, 分数: {top_student['score']}")
    
    # 7. 数据操作
    print("\n=== 7. 数据操作 ===")
    # 添加新列
    csv_df['等级'] = csv_df['score'].apply(lambda x: '优秀' if x >= 90 else '良好')
    print("添加'等级'列后的DataFrame:")
    print(csv_df)
    
    # 删除列
    # csv_df = csv_df.drop('等级', axis=1)

except FileNotFoundError:
    print("错误: 'students.csv' 文件未找到。请确保该文件与脚本在同一目录下。")
