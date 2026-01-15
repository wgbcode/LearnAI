
"""
知识点：
1、将问答数据保存到 SQLite 或其它关系型数据库中
2、支持历史聊天记录功能和上下文摘要生成功能
3、支持多模态，同时支持文本、语音、图片和视频
4、全栈开发，前端 Vue3，接口 FastAPI
"""

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from llm_libs import llm
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import TypedDict

prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个乐于助人的助手。尽你所能回答所有问题。提供的聊天历史包含与你对话用户的相关信息。'),
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    ('human', '{input}')
])

chain = prompt | llm


def get_session_history(session_id: str):
    """从关系型数据库的历史消息列表中 返回当前会话 的所有历史消息"""
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string='sqlite:///chat_history.db',
    )


chain_with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
)


def summarize_messages(current_input):
    """剪辑和摘要上下午，历史记录"""
    session_id = current_input['config']["configurable"]["session_id"]
    if not session_id:
        raise ValueError("必须通过config参数提供session_id")

    # 获取当前会话ID的所有历史聊天记录
    chat_history = get_session_history(session_id)
    stored_messages = chat_history.messages

    if len(stored_messages) <= 2:  # 保留最近2条消息的阈值
        return {"original_messages": stored_messages, "summary": None}  # 不满足摘要条件时返回原始消息

    # 剪辑消息列表
    last_two_messages = stored_messages[-2:]  # 保留的2条消息
    messages_to_summarize = stored_messages[:-2]  # 需要进行摘要的消息列表

    summarization_prompt = ChatPromptTemplate.from_messages([
        ("system", "请将以下对话历史压缩为一条保留关键信息的摘要消息。"),
        ("placeholder", "{chat_history}"),
        ("human", "请生成包含上述对话核心内容的摘要，保留重要事实和决策。")
    ])
    summarization_chain = summarization_prompt | llm
    # 生成摘要(AIMessage)
    summary_message = summarization_chain.invoke({'chat_history': messages_to_summarize})

    # 返回结构化结果（不调用chat_history.clear()）
    return {
        "original_messages": last_two_messages,  # 保留的原始消息
        "summary": summary_message  # 生成的摘要
    }


final_chain = (RunnablePassthrough.assign(messages_summarized=summarize_messages)
               | RunnablePassthrough.assign(
            input=lambda x: x['input'],
            chat_history=lambda x: x['messages_summarized']['original_messages'],
            system_message=lambda
                x: f"你是一个乐于助人的助手。尽你所能回答所有问题。摘要：{x['messages_summarized']['summary'].content}" if
            x[
                'messages_summarized'].get("summary") else "无摘要"
        ) | chain_with_message_history)

# result = final_chain.invoke(
#     {'input': '我的名字叫张三', "config": {"configurable": {"session_id": "user123"}}},
#     config={"configurable": {"session_id": "user123"}})
# print(result)

# 创建 fastAPI 的应用
app = FastAPI(title='多模态聊天机器人', version='1.0.0', description='支持上下文和流式输出功能')


class ParamsType(TypedDict):
    content: str


class ResponseType(TypedDict):
    data: str


@app.post("/v1/chat")
def chat(params: ParamsType) -> ResponseType:
    config = {'configurable': {'session_id': 'zs123'}}
    query_obj = {'input': params.get('content', ''), "config": config}
    resp = final_chain.invoke(query_obj, config=config)
    return {
        "data": resp.content
    }


# 启动服务
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
