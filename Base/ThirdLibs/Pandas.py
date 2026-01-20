import os

# 学习文档：https://pandas.liuzaoqi.com/doc/chapter0/10%E5%88%86%E9%92%9F%E5%85%A5%E9%97%A8pandas.html

import numpy as np
import pandas as pd

print("\n" + "=" * 50)
print("===1、创建数据===")
print("=" * 50)
print(">>（1）第一种创建 DataFrame 的方法")
s = pd.Series([1,2,3,4,5,np.nan,7,8,9])
print(f"Series 方法：\n{s}")
dates = pd.date_range(start="2025-01-01", periods=6)
print(f"date_range 方法：\n{dates}")
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list("ABCD"))
print(f"DataFrame 数据:\n{df}")

print("\n>>（2）第二种创建 DataFrame 的方法")
df2 = pd.DataFrame({
    'A':1,
    'B':pd.Timestamp('2020-01-01'),
    'C':pd.Series(1,index=list(range(4)),dtype='float32'),
    'D':np.array([3]*4,dtype='int32'),
    'E':pd.Categorical(['test','test','test','test']),
    'F':'foo'
})
print(f"DataFrame 数据2:\n{df2}")
print(f"每行数据的类型：\n{df2.dtypes}")


print("\n" + "=" * 50)
print("===2、数据查看===")
print("=" * 50)
print(f"头部两行：\n{df.head(2)}")
print(f"尾部两行：\n{df.tail(2)}")
print(f"索引：\n{df.index}")
print(f"列名\n{df.columns}")
print(f"值：\n{df.values}")
print(f"描述性统计：\n{df.describe()}")
print(f"数据转置（x轴和y轴）：\n{df.T}")
print(f"行排序(倒序）：\n{df.sort_index(axis=0,ascending=False)}")
print(f"列排序(倒序）：\n{df.sort_index(axis=1,ascending=False)}")
print(f"根据B列的值排序(倒序）：\n{df.sort_values(by='B',ascending=False)}")


print("\n" + "=" * 50)
print("===3、数据选取===")
print("=" * 50)
print(">>(1)通过[]选取数据")
print(f"选取A列：\n{df['A']}")
print(f"选取前三行：\n{df[0:3]}")
print(f"按行日期选取：\n{df['2025.01.03':'2025.01.05']}")

print(">>(2)通过标签选取数据")
print(f"df：\n{df}")
print(f"第一行：\n{df.loc[dates[0]]}")
print(f"A列和B列：\n{df.loc[:,'A':'B']}")
print(f"A列和B列（加上日期筛选）：\n{df.loc['2025.1.3':'2025.1.5','A':'B']}")
print(f"A列和B列（一行）：\n{df.loc['2025.1.3','A':'B']}")
print(f"一个值：\n{df.loc['2025.1.3','B']}")

print(">>(3)通过位置选取数据")
print(f"df：\n{df}")
print(f"前三行：\n{df.iloc[3]}")
print(f"同时筛选行和列(连续）：\n{df.iloc[3:5,2:4]}")
print(f"同时筛选行和列(不连续）：\n{df.iloc[[1,2,4],[0,2]]}")
print(f"一个值：\n{df.iloc[1,1]}")

print(">>(4)通过布尔选取数据（可判断）")
print(f"df：\n{df}")
print(f"A列大于0：\n{df[df.A>0]}")
print(f"所有大于0(不满足的会展示为NaN）：\n{df[df>0]}")
df2 = df.copy()
df2['E'] = ['one','two','three','one','one','four']
print(f"添加E列后：\n{df2}")
print(f"根据E列筛选中满足条件的数据：\n{df2[df2['E'].isin(['one','two'])]}")


print("\n" + "=" * 50)
print("===4、缺失值处理===")
print("=" * 50)
df4 = df.reindex(index=dates[0:4],columns=list(df.columns)+['E'])
print(f"通过 reindex 方法拿到一个可操作的实例：\n{df4}")
df4.loc[dates[0]:dates[1],'E'] = 1
print(f"将部分NaN换成1：\n{df4}")
print("开始进行关于NaN的操作")
print(f"涉及到NaN的行不展示：\n{df4.dropna(how='any')}")
print(f"缺失值填充，统一改要数值5：\n{df4.fillna(value=5)}")
print(f"判断单元格数值是否为缺失值：\n{df4.isnull()}")


print("\n" + "=" * 50)
print("===5、常用操作===")
print("=" * 50)
print(">>(1)统计")
df5 = df.dropna(how='any')
print(f"统计之前，先排队缺失值：\n{df5}")
print(f"纵向求平均值：\n{df5.mean()}")
print(f"横向求平均值：\n{df5.mean(1)}")

print(">>(2)Apply函数")
print(f"求列的累积和：\n{df5.apply(np.cumsum)}")
print(f"计算列最大值和最小值的差值：\n{df5.apply(lambda x: x.max()-x.min())}")

print(">>(3)value_counts函数")
df5_3 = pd.Series(np.random.randint(0,7,size=10))
print(f"生成的数据：\n{df5_3}")
print(f"统计相同数值出现的次数：\n{df5_3.value_counts()}")

print(">>(4)字符串方法")
df5_4 = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
print(f"大写变小写：\n{df5_4.str.lower()}")



print("\n" + "=" * 50)
print("===6、数据合并===")
print("=" * 50)
df6 = pd.DataFrame(np.random.randn(10, 4))
print(f"生成的数据：\n{df6}")
pieces = [df6[:2],df6[3:5],df6[8:]]
print(f"数据合并：\n{pd.concat(pieces)}")


print("\n" + "=" * 50)
print("===7、数据分组===")
print("=" * 50)
df7 = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
                          'foo', 'bar', 'foo', 'foo'],
                   'B' : ['one', 'one', 'two', 'three',
                           'two', 'two', 'one', 'three'],
                   'C' : np.random.randn(8),
                   'D' : np.random.randn(8)})
print(df7)
print(f"根据A列分组并求和：\n{df7.groupby('A').sum()}")
print(f"根据A列和B列分组并求和：\n{df7.groupby(['A','B']).sum()}")


print("\n" + "=" * 50)
print("===8、数据可视化===")
print("=" * 50)
df8 = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
print(df8.plot())



print("\n" + "=" * 50)
print("===9、导入和导出数据===")
print("=" * 50)
df9 = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
print("导出数据为 csv 文件")
df9.to_csv('pandas_df9_test.csv')
print(f"从 csv 文件中读取数据:\n{pd.read_csv('pandas_df9_test.csv').head()}")
print("将数据导出为 hdf 文件(二进制，需要安装 tables 库）")
df9.to_hdf('pandas_df9_test_hdf.h5', key='df9_test')
print(f"从 hdf 文件中读取数据：:\n{pd.read_hdf('pandas_df9_test_hdf.h5').head()}")
print("将数据导出为 xlsx 文件(需要安装 openpyxl 库）")
df9.to_excel('pandas_df9_test_xlsx.xlsx')
print(f"从 xlsx 文件中读取数据：:\n{pd.read_excel('pandas_df9_test_xlsx.xlsx').head()}")


















