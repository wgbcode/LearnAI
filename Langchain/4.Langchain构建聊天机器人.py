
# 第三方依赖包：pip install "langserve[all]"
# 接口测试工具：Postman； ApiPost
# 第三方依赖包：pip install langchain_community

import os

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory

# 聊天机器人案例
model = ChatOpenAI(model='gpt-4-turbo')

# 定义提示模板
prompt_template = ChatPromptTemplate.from_message([
    ('system','你是一个乐于助人的助手。用{language}尽你所能回答所有问题'),
    MessagesPlaceholder(variable_name='my_msg') # 添加历史记录，注释掉，每次都是单独的
])

# 得到链
chain = prompt_template | model

# 保存聊天的历史记录
store = {} # 所有用户的聊天记录都保存到store。key：sessionId,value：历史聊天记录对象

#  此函数预期将接收一个 session_id 并返回一个历史消息记录对象
def get_session_history(session_id:str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='my_msg' # 每次聊天时发送 msg 的 key
)

config = {'configurable':{'session_id':'zs123'}} # 给当前会话定义一个 sessionId

# 第一轮
resp = do_message.invoke(
    {
        'my_msg':[HumanMessage(content='你好啊！我是LaoXiao')],
        'language':'中文'
    },
    config=config
)

print(resp.content)

# 第二轮：返回的数据基于上一轮的问答
resp2 = do_message.invoke(
    {
        'my_msg':[HumanMessage(content='请问我的名字是什么？')],
        'language':'中文'
    },
    config=config
)

print(resp2.content)


# 第三轮：返回的数据是流式的
config = {'configurable':{'session_id':'zs1234'}} # 不换 session_id，英文不生效
resp3 = do_message.stream(
    {
        'my_msg':[HumanMessage(content='请给我讲一个笑话？')],
        'language':'英文'
    },
    config=config
)
for resp in resp3:
    # 每一次resp都是一个token
    print(resp.content,end='-')















