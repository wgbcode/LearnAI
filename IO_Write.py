### 在硬盘中写


# 写入一行 Hello World。w：如果文件存在，会覆盖；a：追加，不会覆盖
# f = open("./IO/test.txt","w",encoding='utf-8')
f = open("./IO/test.txt","a",encoding='utf-8')
f.write("Hello World(write)\n")
f.close()


# 使用 with
with open("./IO/test.txt","r",encoding='utf-8') as f:
    print('写入（with):', f.read())
    print(f.read())