
# 第三方依赖包：pip install "langserve[all]"
# 接口测试工具：Postman； ApiPost
# 第三方依赖包：pip install langchain_community

import uvicorn
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langserve import add_routes
from typing import TypedDict
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

# 加载环境变量
load_dotenv()

# 聊天机器人案例
model = ChatOpenAI(model='gpt-5-mini')

# 定义提示模板
prompt_template = ChatPromptTemplate.from_messages([
    ('system','你是一个乐于助人的助手。用{language}尽你所能回答所有问题'),
    MessagesPlaceholder(variable_name='my_msg') # 添加历史记录，注释掉，每次都是单独的
])

# 得到链。能够通过链连接起来的，都是 Runnable 对象
chain = prompt_template | model

# 保存聊天的历史记录
store = {} # 所有用户的聊天记录都保存到store。key：sessionId,value：历史聊天记录对象

#  此函数预期将接收一个 session_id 并返回一个历史消息记录对象
def get_session_history(session_id:str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 包装链
do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='my_msg' # 每次聊天时发送 msg 的 key
)

# 创建 fastAPI 的应用
app = FastAPI(title='聊天机器人',version='1.0.0',description='支持上下文功能和流式输出功能')

# 自定义接口
class ParamsType(TypedDict):
    content: str
class ResponseType(TypedDict):
    data: str

@app.post("/v1/chat")
def chat(params: ParamsType)->ResponseType:
    config = {'configurable': {'session_id': 'zs123'}}  # 给当前会话定义一个 sessionId
    resp = do_message.invoke(
        {
            'my_msg':[HumanMessage(content=params.get('content'))],
            'language':'中文'
        },
        config=config
    )
    return {
        "data":resp.content
    }

@app.post("/v1/stream/chat")
def chat_stream_sync(params: ParamsType):
    config = {'configurable': {'session_id': 'zs12345'}}

    def generate():
        data_stream = do_message.stream(
            {
                'my_msg': [HumanMessage(content=params.get('content', ''))],
                'language': '中文'
            },
            config=config
        )

        for resp in data_stream:
            if hasattr(resp, 'content') and resp.content:
                yield f"{json.dumps({'data': resp.content}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

# 启动服务
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
















