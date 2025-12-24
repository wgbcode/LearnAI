### 在硬盘中读

### 适用：文件很小，使用 read()
# 写法一
try:
    f = open("./IO/test.txt", "r")
    print(f"获取文件的内容(写法一）：\n{f.read()}")
finally:
    if f:
        f.close()


# 写法二（简化；自动关闭 close）
with open("./IO/test.txt", "r") as f:
    print(f"获取文件的内容（写法2）：\n{f.read()}")



### 适用：配置文件，使用 readlines()，需要指定正确的编码格式
with open("./IO/package.json", "r", encoding='utf-8') as f:
    print('配置文件:')
    for line in f.readlines():
        print(line.strip())



### 适用：大文件，使用 read(size)，要不内存可能会爆
with open("./IO/big.txt", "r") as f:
    print(f'大文件:\n{f.read(100)}')


### 适用：二进制文件
with open("./IO/rb.png", 'rb') as f:
    print(f"二进制文件:\n{f.read()}")

### 忽视编辑错误
with open("./IO/test.txt", "r", encoding='gbk', errors='ignore') as f:
    print(f"忽视编码错误：\n{f.read()}")