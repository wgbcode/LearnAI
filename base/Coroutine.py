import asyncio

### 1、示例一：生产者函数和消费者函数

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
    print("完成所有数据的生产和消费\n\n\n")

# 获取迭代器实例
print('准备获取迭代器实例')
c = consumer()
print('已经获取迭代器实例\n')
# 开始生产，将迭代器实例传进去
producer(c)



### 示例二：使用 async 和 await 实现协程
# 使用 async...await 会创建一个协程，本质是一个迭代器
#（1）单个 async...await
async def hello(name):
    print(f"hello {name}!!!(会休眠一秒）")
    # sleep 可以换成真正的 IO 操作，那么，多个 IO 操作，就可以在同一线程中并发执行
    await asyncio.sleep(1)
    print(f"hello {name} again!!!")
print("准备使用 run 方法执行单个协程")
asyncio.run(hello('Tom'))
print("\n\n\n")

#（2）多个 async...await
print("准备使用 gather 方法多个协程")
async def mul_coroutine():
    await asyncio.gather(hello('Jack'), hello('Jill'))
asyncio.run(mul_coroutine())



