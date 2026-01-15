import time

from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.tracers import Run


def test1(x: int):
    return x + 10


# 节点： 标准：Runnable
r1 = RunnableLambda(test1)  # 把行数封装成一个组件


# res = r1.invoke(4)

# 2、批量调用
# res = r1.batch([4, 5])

# 3、流式调用
def test2(prompt: str):
    for item in prompt.split(' '):
        yield item


# r1 = RunnableLambda(test2)  # 把行数封装成一个组件
# res = r1.stream('This is a Dog.')  # 返回的是一个生成器
#
# for chunk in res:
#     print(chunk)


#  4、 组合链
r1 = RunnableLambda(test1)
r2 = RunnableLambda(lambda x: x * 2)

chain1 = r1 | r2  # 串行
# print(chain.invoke(2))

#  5、 并行运行
chain = RunnableParallel(r1=r1, r2=r2)
# max_concurrency: 最大并发数
# print(chain.invoke(2, config={'max_concurrency': 3}))

# new_chain = chain1 | chain
# new_chain.get_graph().print_ascii() # 打印链的图像描述
# print(new_chain.invoke(2))

# 6、合并输入，并处理中间数据
# RunnablePassthrough： 允许传递输入数据，可以保持不变或添加额外的键。 必须传入一个字典数据， 还可以过滤

r1 = RunnableLambda(lambda x: {'key1': x})
r2 = RunnableLambda(lambda x: x['key1'] + 10)
r3 = RunnableLambda(lambda x: x['new_key']['key2'])

# chain = r1 | RunnablePassthrough.assign(new_key=r2)  # new_key, 随意定制的，代表输出的key
# chain = r1 | RunnablePassthrough() | RunnablePassthrough.assign(new_key=r2)  # new_key, 随意定制的，代表输出的key
# chain = r1 | RunnableParallel(foo=RunnablePassthrough(), new_key=RunnablePassthrough.assign(key2=r2))
# chain = r1 | RunnableParallel(foo=RunnablePassthrough(), new_key=RunnablePassthrough.assign(key2=r2)) | RunnablePassthrough().pick(['new_key']) | r3
# print(chain.invoke(2))

# 7、后备选项 ： 后备选项是一种可以在紧急情况下使用的替代方案。
r1 = RunnableLambda(test1)
r2 = RunnableLambda(lambda x: int(x) + 20)
# 在加法计算中的后备选项
chain = r1.with_fallbacks([r2])  # r2是r1的后备方案， r1报错的情况下
# print(chain.invoke('2'))

# 8、如有报错，重复多次执行某个节点
counter = -1  # 计数用的


def test3(x):
    global counter
    counter += 1
    print(f'执行了 {counter} 次')
    return x / counter


r1 = RunnableLambda(test3).with_retry(stop_after_attempt=4)
# print(r1.invoke(2))

# 根据条件，动态的构建链
r1 = RunnableLambda(test1)
r2 = RunnableLambda(lambda x: [x] * 2)


# 根据r1的输出结果，判断，是否要执行r2，（判断本身也是一个节点）
# chain = r1 | RunnableLambda(lambda x: r2 if x > 12 else RunnableLambda(lambda x: x))
# print(chain.invoke(1))


# 生命周期管理

def test4(n: int):
    time.sleep(n)
    return n * 2


r1 = RunnableLambda(test4)


def on_start(run_obj: Run):
    """ 当r1节点启动的时候，自动调用"""
    print('r1启动的时间： ', run_obj.start_time)


def on_end(run_obj: Run):
    """ 当r1节已经运行结束的时候，自动调用"""
    print('r1结束的时间： ', run_obj.end_time)


chain = r1.with_listeners(on_start=on_start, on_end=on_end)
print(chain.invoke(2))
