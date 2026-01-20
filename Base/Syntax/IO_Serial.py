import pickle
import json

### 问：为什么要序列化？
### 答：序列化后的数据(为二进制数据），才能保存到硬盘中，否则只能保存到内存中，保存在内存中的数据无法持久化

### 问：pickle 序列化和 JSON 序列化的区别
### 答：pickle 序列化只适合特定版本的 Syntax，JSON 序列化适合所有编辑语言

### 1、使用 pickle 序列化
# 使用 dumps 获取序列化的值
d = dict(name='Bob',age=18)
print(f"序列化后的数据(pickle)：{pickle.dumps(d)}")
"""
使用 dump 序列化数据和保存数据到特定文件
- 第二个参数使用 wb
- 使用 open 方法时，如果文件不存在，会自动创建
- 每次执行时，文件都会被覆盖
"""
with open("./IO/dump.txt","wb") as f:
    pickle.dump(d, f)
# 查看写入 dump.txt 的数据
with open("./IO/dump.txt","rb") as f:
    d = pickle.load(f)
    print(f"写入 dump.txt 中的数据(pickle)：{d}")


### 2、使用 JSON 序列化和反序列化普通数据
# 使用 json 拿到返回的序列化数据
d = dict(name='Jack',age=66)
print(f"序列化后的数据(json)：{json.dumps(d)}")
# 获取反序列化后的数据
json_str = '{"name": "Tom", "age": 19}'
print(f"反序列化后的数据(json)：{json.loads(json_str)}")


### 3、使用 JSON 序列化和反序列化类。直接序列化和反序列化会报错，需要定义一个转换函数
class Student(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age

student = Student("Tom", 19)

# 方法一：使用自定义的转换函数
def student2dict(s):
    return {"name": s.name, "age": s.age}
def dict2student(d):
    return Student(d["name"], d["age"])
dumpDict = json.dumps(student,default=student2dict)
print(f"序列化后的类(josn-class)：{dumpDict}")
json_str = '{"name": "Tom", "age": 19}'
loadDict = json.loads(json_str,object_hook=dict2student)
print(f"反序列化后的类(josn-class)：{loadDict}")


