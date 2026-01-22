import pandas as pd
from tools import old_db,new_db
import shutil
import sqlite3

def update_db():
    """
    函数的作用：
    1、复制一个新的数据库
    2、将数据库中的时间替换成最新的时间
    """

    # 一、复制一份数据库，防止误操作
    shutil.copy(old_db, new_db)

    # 二、数据库操作准备工作
    conn = sqlite3.connect(new_db)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn).name.tolist()
    tdf = {}
    for table in tables:
        tdf[table] = pd.read_sql(f"SELECT * FROM {table}", conn)

    # 三、时间替换
    # 找到时间差值
    example_time = pd.to_datetime(tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)).max()
    current_time = pd.to_datetime("now").tz_localize(example_time.tz)
    time_diff = current_time - example_time

    # 更新bookings表中的book_date
    tdf["bookings"]["book_date"] = (
            pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True) + time_diff
    )

    # 需要更新的日期列
    datetime_columns = ["scheduled_departure", "scheduled_arrival", "actual_departure", "actual_arrival"]
    for column in datetime_columns:
        tdf["flights"][column] = (
                pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff
        )

    # 将更新后的数据写回数据库
    for table_name, df in tdf.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        del df  # 清理内存
    del tdf  # 清理内存

    conn.commit()
    conn.close()

    return new_db


if __name__ == '__main__':
    update_db()