"""
LCEL的各种语法
1.Runnable节点
2.节点调用、批量、流式运行
3.组合成chain
4.并行调用运行
5.合并输入和输出字典
6.后备选项
7.重复多次执行Runnable节点
8.条件构建chain
9.map高阶处理
10.打印chain图形
11.生命周期管理
"""
from langchain_core.tracers import Run
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

# 1、串行
r1 = RunnableLambda(lambda x: x + 1)
r2 = RunnableLambda(lambda x: x * 2)
chain1 = r1 | r2
print(chain1.invoke(2))

# 2、RunnableParallel：并行
chain2 = RunnableParallel(r1=r1, r2=r2)
# max_concurrency:最大并发数
print(chain2.invoke(2,config={'max_concurrency':2}))

# 3、RunnablePassthrough：根据需求新增、删除、筛选 key
# 不合并输入时，只有输出
r3 = RunnableLambda(lambda x: {'input': x})
r4 = RunnableLambda(lambda x: x['input'] + 10)
chain3 = r3 | r4
print(chain3.invoke(2))
# 合并输入后，同时有输入和输出。output 是随意定义的 key，起任何名都可以
chain4 = r3 | RunnablePassthrough.assign(output=r4)
print(chain4.invoke(2))
# 一个复杂点的例子。RunnablePassthrough 如果什么都不传，就会直接拿到上一个链的结果
chain5 = r3 | RunnableParallel(
    test_key1=RunnablePassthrough(),
    test_key2=RunnablePassthrough.assign(output=r4),
)
print(chain5.invoke(2))
# 再加一层过滤，只需要留下 test_key2
chain6 =  RunnablePassthrough.pick(keys=['test_key2'],self=chain5)
print(chain6.invoke(2))

# 4、with_fallbacks：后备选项。紧急情况下使用，如断网和报错
r5 = RunnableLambda(lambda x: x + 1)
r6 = RunnableLambda(lambda x: int(x) * 2)
# r5 如果报错，就执行 r6
chain7 = r5.with_fallbacks([r6])
print(chain7.invoke('2'))

# 5、根据条件动态组件合链。如通过这个方法写一个路由调度器，针对不同领域的问题，使用不同的大模型调用链
r7 = RunnableLambda(lambda x: x + 1)
r8 = RunnableLambda(lambda x: x * 20)
r9 = RunnableLambda(lambda x: x * 30)
chain8 = r7 | RunnableLambda(lambda x: r8 if x % 2 == 0 else r9)
print(chain8.invoke(1))
print(chain8.invoke(2))

# 6、with_listeners: 生命周期
r10 = RunnableLambda(lambda x: x + 1)
def on_start(run_obj:Run):
    """当 r10 节点启动的时候，自动调用"""
    print(f'r10 节点启动的时间：{run_obj.start_time}')
def on_end(run_obj:Run):
    """当 r10 节点结束的时候，自动调用"""
    print(f'r10 节点结束的时间： {run_obj.end_time}' )
chain9 = r10.with_listeners(on_start=on_start, on_end=on_end)
print(chain9.invoke(2))










