### 在内存中读写

### 读写内存中的字符串和二进制
from io import BytesIO,StringIO

# 写入字符串1（字符在初始化后，再逐个传入）
f = StringIO()
f.write("Hello")
f.write(" ")
f.write("world")
# 获取写入的字符串的值
print(f"写入的字符串为：{f.getvalue()}")

# 写入字符串2（字符在初始化后，再逐个传入）
f1 = StringIO()
f1.write("Hello123")
f1.write(" ")
f1.write("world123")
# 获取写入的字符串的值
print(f"写入的字符串为：{f1.getvalue()}")
# 从0处开始截断，可达到清空的效果
f1.truncate(0)
print(f"截断后字符串的值为：{f1.getvalue()}")


# 逐行读取（字符在初始化 StringIO 时，就传入）
print("\n\n初始化就传入")
f = StringIO("Hello!\nHi!\nByeBye boy")
while True:
    s = f.readline()
    if s == '':
        break
    print(s)


### 二进制数据写入内存和获取写入的值
# 方式一。写入的值需要是一个 bytes-like object
f = BytesIO()
f.write("中文".encode("utf-8"))
print(f"写入的二进制{f.getvalue()}")

# 方式二。字符串前需要加 b
f = BytesIO(b'\xe4\xb8\xad\xe6\x96\x87')
print(f"写入的二进制:{f.read().decode('utf-8')}")



