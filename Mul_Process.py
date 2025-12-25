import os,time,random
from multiprocessing import Process,Pool

"""
关于 fork 方法：
1、os.fork() 方法会创建一个子进程，但只能在 linux/mac 系统上使用，在 window 上无法使用
2、fork 调用一次，会返回两次，父进程永远返回进程 id，子进程永远返回 0
3、子进程可以通过 os.fork().getppid() 拿到父进程的 id
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
    p.close()
    print("等待所有子进程执行完毕")
    p.join()
    print("所有子进程都已执行结束")



