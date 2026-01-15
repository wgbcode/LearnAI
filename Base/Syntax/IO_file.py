import os

### 1、获取操作系统名字，posix 是 linux/macos/unix，nt 是 windows
print(f"当前操作系统：{os.name}")

### 2、获取详细的系统信息(uname 函数在 window 上不提供)
# print(f"当前操作系统详细系统信息：{os.uname()}")

### 3、获取操作系统的环境变量
print(f"环境变量：{os.environ}")
print(f"环境变量(Path)：{os.environ['PATH']}")
print(f"环境变量(Path)：{os.environ.get('PATH')}")
path_list = os.environ["PATH"].split(os.pathsep)
print("优雅查看")
for path in path_list:
    print(path)

### 4、操作文件和目录
# 获取当前目录的绝对路径
abspath = os.path.abspath('..')
print(f"当前目录的绝对路径：{abspath}")
# 使用 join 生成新的绝对路径(使用 join 的原因：不同的系统，会匹配不同的路径分隔符；同理，拆分也使用 split）
pathAfterJoin = os.path.join(abspath, 'testDir')
# 创建一个目录
os.mkdir(pathAfterJoin)
# 删除一个目录
os.rmdir(pathAfterJoin)
# 使用 split 拆分。返回值：（最后级别目录, 文件名）
pathAfterJoin2 = os.path.join(abspath, 'IO/rb.png')
print(f"路径拆分后的返回值：{os.path.split(pathAfterJoin2)}")
# 使用 splitext 获取文件扩展名
print(f"文件扩展名：{os.path.splitext(pathAfterJoin2)}")
# 对文件重命名
pathAfterJoin3 = os.path.join(abspath, 'IO/rb3.png')
pathAfterJoin4 = os.path.join(abspath, 'IO/rb4.png')
# os.rename(pathAfterJoin3, pathAfterJoin4)
# 删除文件
# os.remove(pathAfterJoin3)
# 列出所有 .py 的文件
pyArr = [x for x in os.listdir('..') if os.path.isfile(x) and os.path.splitext(x)[1] == '.py']
print(f"所有 .py 的文件：{pyArr}")
