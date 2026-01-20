import os,time,random
from multiprocessing import Process,Pool,Queue
import subprocess

"""
关于 fork 方法：
1、os.fork() 方法会创建一个子进程，但只能在 linux/mac 系统上使用，在 window 上无法使用
2、fork 调用一次，会返回两次，父进程永远返回进程 id，子进程永远返回 0
3、子进程可以通过 os.fork().getppid() 拿到父进程的 id
"""

"""
关于分布式进程：
1、学习文档：https://liaoxuefeng.com/books/python/process-thread/process-manager/index.html
2、原理1：在机器一上创建 10 个进程，然后经过 QueueManager 进行封装，并通过网格接口将 QueueManager 实例发送到机器二
3、原理2：机器一和机器二，都通过  QueueManager 实例 get_task_queue() 和 get_result_queue() 两个方法拿到进程实例，并进行业务操作
"""


# 定义一个子进程的回调函数，它包括子进程需要执行的业务代码
def run_process(name):
    print(f"子进程执行了，名字：{name}，ID:{os.getpid()}")

# 定义子进程的回调函数（多个子进程共用）
def run_process2(name):
    print(f"开始执行子进程回调, name:{name}，ID:{os.getpid()}")
    start = time.time()
    time.sleep(random.random()*3)
    end = time.time()
    print(f"子进程执行结束，name:{name}, 时间：{end-start}")


# 写数据进程执行的代码:
def write(q):
    print('Process to write: %s' % os.getpid())
    for value in ['A', 'B', 'C']:
        print('Put %s to queue...' % value)
        q.put(value)
        time.sleep(random.random())

# 读数据进程执行的代码:
def read(q):
    print('Process to read: %s' % os.getpid())
    while True:
        value = q.get(True)
        print('Get %s from queue.' % value)

"""
1、作为脚本执行：__name__ == '__main__' 中的代码只会在执行脚本，即 python Mul_Process.py 时，才会执行
2、作为模块导入：在导入模块时，即 import Mul_Process 时，不会执行
3、主进程作用域和子进程作用域：表示的是主进程。p.start 会创建一个 child 的子进程，后面的代码也是在 child 的子进程中执行。如果 run_process 定义
在主进程 __main__ 内，就无法找到函数，然后报错。__name__ == '__main__' 这一行是必须要加的，因为创建子进程，必须在主进程内，才能创建
"""
if __name__ == '__main__':  # ✅ 关键：保护入口代码
    ### 1、使用 Process 创建一个子进程
    print(f"Process：创建一个子进程实例，ID:{os.getpid()}")
    p = Process(target=run_process, args=("child",))
    print("开始启动子进程(start)")
    p.start()
    print("等待子进程结束后，再继续往下执行（join)")
    p.join()
    print("子进程执行结束\n\n\n")

    ### 2、使用 Pool 创建多个子进程
    print(f"Pool：创建多个子进程实例，ID:{os.getpid()}")
    p=Pool(4)
    for i in range(5):
        p.apply_async(run_process2, args=(i,))
    print("调用 close 方法，禁止继续创建子进程")
    p.close()
    print("等待所有子进程执行完毕")
    p.join()
    print("所有子进程都已执行结束\n\n\n")

    ### 3、使用 subprocess 模块执行命令行（外部进程）
    print("$ nslookup www.python.org")
    r = subprocess.run("nslookup www.python.org", shell=True)
    print("Exit code:", r.returncode)
    print("\n\n\n")


    ### 4、进程间通讯。在 main 中创建三个子进程（queue、read、write），read 进程从 queue 进程读数据，write 从 queue 写数据
    print("Queue: 创建 q、pw、pr 三个子进程")
    q = Queue()
    pw = Process(target=write, args=(q,))
    pr = Process(target=read, args=(q,))
    print("启动 pw 进程，写入")
    pw.start()
    print("启动 pr 进程，读取")
    pr.start()
    print('等待 pw 结束')
    pw.join()
    print('pr 是死进程，会无限循环，只能强行结束')
    pr.terminate()



