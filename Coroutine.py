# 消费者函数
# 首次执行时，会返回一个迭代器对象，指针停在函数开头
def consumer():
    print('start[只执行一次；send(None)和next；到达第一个 yield]\n')
    while True:
        # 第一次，需要运行 next 或者 send(None) 时，才会执行到这里
        x = yield
        print(f"消费者：消费了数字 {x}")
        if not x:
            # 边界：若返回空值，直接结束
            return
        print('end（无限循环，又回到 yield，等待下一个 send）\n')

# 生产者函数
# 实现的功能：生产者函数生产一个数字，消费者执行一次，直到调用 close 方法结束
# 传入的参数是迭代器对象，需要通过这个对象调用 send、close 方法
def producer(c):
    # 跳转到第一个 yield 处，只需要执行一次
    c.send(None) # 或者 c.next()
    n = 0
    while n < 5:
        n += 1
        print(f"生产者：生产了数字 {n}")
        c.send(n)
    c.close()
    print("完成所有数据的生产和消费")

# 获取迭代器实例
print('准备获取迭代器实例')
c = consumer()
print('已经获取迭代器实例\n')
# 开始生产，将迭代器实例传进去
producer(c)




