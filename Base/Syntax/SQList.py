import os
# SQLite 默认内置
import sqlite3
# 连接到 wgb_test.db 数据库，如果没有，会创建
conn = sqlite3.connect('wgb_test.db')
# 查看数据库的位置
db_path = os.path.abspath('wgb_test.db')
print(f"数据库文件路径：{db_path}")
# 创建一个游标，用于执行 SQL 命令和获取查询结果
cursor = conn.cursor()
# 执行 SQL 语句
cursor.execute("create table if not exists user (id INTEGER primary key AUTOINCREMENT, name varchar(20), password varchar(20))")
# 插入一条语句
cursor.execute("insert into user (name, password) values (\'wgb\',\'123456\')")
# 查询所有的数据
cursor.execute("select * from user")
# 获取所有查询结果
rows = cursor.fetchall()
print("打印查询结果")
print("ID\tName\tPassword")
for row in rows:
    print(row)
# 提交事务
conn.commit()
# 关闭游标
cursor.close()
# 关闭数据库连接
conn.close()

