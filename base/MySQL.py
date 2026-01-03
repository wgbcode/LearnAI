# 导入 MySQL 驱动
# 依赖包安装  pip install mysql-connector-python
import mysql.connector

# 判断 test 数据库是否存在，如果不存在，则创建。有异常，则抛出
# def create_database_if_not_exist(cursor,database_name):
#     try:
#         cursor.execute(f"create database if not exists {database_name}")
#         print(f"数据库{database_name}已准备就绪")
#     except mysql.connector.Error as error:
#         print("创建数据库失败:{}".format(error))
#         raise

# 创建连接实例
# MySQL 官网：https://dev.mysql.com/downloads/mysql/
# MySQL 安装教程：https://liaoxuefeng.com/books/python/database/mysql/index.html
# 默认数据库 test_db_mysql 存在，如果不存在，要先手动创建
conn = mysql.connector.connect(user='root',password='password',database='test_db_mysql')
# 创建游标
cursor = conn.cursor()
# 判断 test_db_mysql 数据库是否存在
# create_database_if_not_exist(cursor,'test_db_mysql')
# 创建 user 表格，并插入数据
cursor.execute("""
    CREATE TABLE IF NOT EXISTS user (
        id INTEGER PRIMARY KEY AUTO_INCREMENT,
        name VARCHAR(20),
        password VARCHAR(20)
    )
""")
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
# 关闭 SQL 连接
conn.close()






