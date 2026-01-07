from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
import gradio as gr

from langchain_demo.my_llm import llm

# 1、提示词模板
prompt = ChatPromptTemplate.from_messages([
    ('system', "{system_message}"),  # 动态注入系统消息
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    ('human', '{input}')
])

chain = prompt | llm  # 基础的执行链


# 2、存储聊天记录： （内存，关系型数据库或者redis数据库）

def get_session_history(session_id: str):
    """从关系型数据库的历史消息列表中 返回当前会话 的所有历史消息"""
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string='sqlite:///chat_history.db',
    )


# langchain所有消息类型： SystemMessage, HumanMessage, AIMessage, ToolMessage


# 3、创建带历史记录功能的处理链

chain_with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
)


# 4、剪辑和摘要上下午，历史记录。 ： 保留最近的前2条消息， 把之前的所有消息形成摘要

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


# 最终的链
# 1、 {input: 原来的，messages_summarized=summarize_messages函数执行后的返回值}
# 2、 {input: 原来的， chat_history: messages_summarized['original_messages'], system_message: messages_summarized['summary_message']}
final_chain = (RunnablePassthrough.assign(messages_summarized=summarize_messages)
               | RunnablePassthrough.assign(
            input=lambda x: x['input'],
            chat_history=lambda x: x['messages_summarized']['original_messages'],
            system_message=lambda
                x: f"你是一个乐于助人的助手。尽你所能回答所有问题。摘要：{x['messages_summarized']['summary'].content}" if x[
                'messages_summarized'].get("summary") else "无摘要"
        ) | chain_with_message_history)

# result1 = final_chain.invoke({'input': '你好，我是蔡锷?', "config": {"configurable": {"session_id": "user123"}}},
#                                             config={"configurable": {"session_id": "user123"}})
# print(result1)
#
# result2 = final_chain.invoke({'input': '我的名字叫什么?', "config": {"configurable": {"session_id": "user123"}}},
#                                             config={"configurable": {"session_id": "user123"}})
# print(result2)

# result3 = final_chain.invoke(
#     {'input': '用我的名字，写一篇短文，不超过200个字？', "config": {"configurable": {"session_id": "user123"}}},
#     config={"configurable": {"session_id": "user123"}})
# print(result3)


#  web界面中的核心函数
def add_message(chat_history, user_message):
    if user_message:
        chat_history.append({"role": "user", "content": user_message})
    return chat_history, gr.Textbox(value=None, interactive=False)


def execute_chain(chat_history):
    input = chat_history[-1]
    result = final_chain.invoke({'input': input['content'], "config": {"configurable": {"session_id": "user123"}}},
                                            config={"configurable": {"session_id": "user123"}})
    chat_history.append({'role': 'assistant', 'content': result.content})
    return chat_history


# 开发一个聊天机器人的Web界面
with gr.Blocks(title='多模态聊天机器人', theme=gr.themes.Soft()) as block:

    # 聊天历史记录的组件
    chatbot = gr.Chatbot(type='messages', height=500, label='聊天机器人')

    with gr.Row():

        # 文字输入的区域
        with gr.Column(scale=4):
            user_input = gr.Textbox(placeholder='请给机器人发送消息...', label='文字输入', max_lines=5)

            submit_btn = gr.Button('发送', variant="primary")

        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=['microphone'], label='语音输入', type='filepath', format='wav')


    chat_msg = user_input.submit(add_message, [chatbot, user_input], [chatbot, user_input])
    chat_msg.then(execute_chain, chatbot, chatbot)



if __name__ == '__main__':
    block.launch()