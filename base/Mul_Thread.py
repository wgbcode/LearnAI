import time,threading


### 1、使用 threading 创建一个线程，并执行业务代码
def loop():
    print(f"开始执行业务代码, 线程名为 {threading.current_thread().name}\n")
    n=0
    while n<5:
        print(f"执行中：{n}")
        time.sleep(1)
        n+=1
    print("业务代码执行结束")
print("准备创建一个线程")
t = threading.Thread(target=loop,name="test_thread")
print("准备调用 start 方法执行线程")
t.start()
print("调用 join 方法，等待线程执行结束")
t.join()
print("线程执行结束\n\n\n")


### 2、多线程修改变量时，要使用锁
"""
1、多进程：每个进程的变量是独立的，修改变量不会受到影响
2、多线程：如果共用同一个全局变量，有可能出现竞态，导致把数据改乱，所以需要锁
"""
# (1)将数据改乱的例子
# 银行存款
balance = 0
def change_it(n):
    # global 的作用：告诉 python，函数内部的 balance 使用的是全局变量，并允许修改
    # 如果不使用 global，python 会认为 balance 是局部变量
    global balance
    balance = balance + n
    balance = balance - n
def run_the_thread(n):
    for i in range(1000000):
        change_it(n)
t1 = threading.Thread(target=run_the_thread,args=(5,))
t2 = threading.Thread(target=run_the_thread,args=(8,))
t1.start()
t2.start()
t1.join()
t2.join()
print(f"线程执行结束后，balance 的值： {balance}")

# (2)上锁
balance2 = 0
# 获取锁的实例
lock = threading.Lock()
def change_it2(n):
    global balance2
    balance2 = balance2 + n
    balance2 = balance2 - n
def run_the_thread2(n):
    for i in range(1000000):
        # 先要获取锁
        lock.acquire()
        try:
            change_it(n)
        finally:
            # 改完后一定记得要释放锁
            lock.release()
t3 = threading.Thread(target=run_the_thread2,args=(15,))
t4 = threading.Thread(target=run_the_thread2,args=(18,))
t3.start()
t4.start()
t3.join()
t4.join()
print(f"线程执行结束后，balance 的值： {balance}\n\n\n")

# （3）不加锁，使用 ThreadLocal
print("创建全局 ThreadLocal 对象，本质是一个字典 dict")
local_school = threading.local()
def process_thread(name):
    local_school.student = name
    print(f"我是线程：{threading.current_thread().name},读取到的值为：{local_school.student}")
t5 = threading.Thread(target=process_thread,args=('Tom',))
t6 = threading.Thread(target=process_thread,args=('John',))
t5.start()
t6.start()
t5.join()
t6.join()


# (4) 补充说明：
"""
假设电脑的四核 CPU
1、开启一个线程，让程序死循环，在任务管理器中，CPU 会占到 100%
2、开启四个线程，让每个线程的程序都死循环，在任务管理器中，CPU 也会是 100% 左右（根本起不到多核的作用，如果是 Java，就可以到 400%)
3、历史遗留原因：在 python 中，每个线程执行时，都会获得 GIL 锁，每执行100条字节码，解释器就自动释放GIL锁，让别的线程有机会执行。这个GIL全局锁实际上把所有线程的执行代码都给上了锁，所以，多线程在Python中只能交替执行，即使100个线程跑在100核CPU上，也只能用到1个核
4、推荐方案（充分利用多核）：多进程 + 协程（Coroutine）
5、协程的原理：一个进程配一个协程，能把多核 CPU 的性能吃满，协程可以执行到一半时中断，可以同时执行多个任务
"""