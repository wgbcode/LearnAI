"""
执行顺序：
1、步骤一：用户输入问题
2、步骤二：RunnableWithMEssageHistory 管理会话历史
3、步骤三：通过 create_retrieval_chain 方法创建链，先执行 history_chain，再执行 main_chain（核心）
4、步骤四：执行 history_chain 链（历史感知检索链），判断问题是否需要重写（核心）
（1）问题一"What is Task Decomposition?"不需要重写
（2）问题二”What are common ways of doing it?"需要重写，主要是重写 it，重写后的问题为："What are common ways of doing task decomposition?"
5、步骤五：检索相关文档。目前是提前写死和检索，并保存到了向量数据库，但也可以由用户传入（核心）
6、步骤六：执行 main_chain 链（文档问答链）。根据 “历史上下文+感知检索内容(重写/非重写）+原始问答（it)” 生成答案（核心）
7、步骤七：LLM 大模型生成答案
8、步骤八：返回最终答案
"""

import bs4
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

load_dotenv()

# 一、创建模型
model = ChatOpenAI(model='gpt-5-mini')

# 二、检索文档（写死）
# 配置 bs_kwargs 的目的：爬虫时，让解析器只解析我想要的内容（特定 class），而不是整个文档
loader = WebBaseLoader(
    web_paths=['https://lilianweng.github.io/posts/2023-06-23-agent/'],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=('post-header', 'post-title', 'post-content'))
    )
)
docs = loader.load()
print(f"文档内容：{docs}")
print(f"文档长度：{len(docs)}")
# 大文本切割。目的：大模型 token 存在上限，文档内容过多时，不能一次性直接传给大模型，会导致报错
# chunk_size 表示每个被切割文本块的大小，chunk_overlap 表示相邻被切割文本块可重叠字符的数量（预防语义丢失）
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)
print(f"被切换后的文档：{splits}")
print(f"被切换后的文档的长度：{len(splits)}")
# 存储到向量数据库
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
# 生成一个检索器，在子链中使用
retriever = vectorstore.as_retriever()

# 三、创建子链
contextualize_q_system_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, 
formulate a standalone question which can be understood 
without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""
retriever_history_temp = ChatPromptTemplate.from_messages(
    [
        ('system', contextualize_q_system_prompt),
        MessagesPlaceholder('chat_history'),
        ("human", "{input}"),
    ]
)
history_chain = create_history_aware_retriever(model, retriever, retriever_history_temp)

# 四、创建主链
main_system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer 
the question. If you don't know the answer, say that you 
don't know. Use three sentences maximum and keep the answer concise.\n

{context}
"""
main_prompt = ChatPromptTemplate.from_messages(  # 提问和回答的 历史记录  模板
    [
        ("system", main_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
main_chain = create_stuff_documents_chain(model, main_prompt)

# 五、把上下文功能加进去，生成一个 Runnable 对象
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
chain = create_retrieval_chain(history_chain, main_chain)
result_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)

# 六、执行
# 第一轮对话
resp1 = result_chain.invoke(
    {'input': 'What is Task Decomposition?'},
    config={'configurable': {'session_id': 'zs123456'}}
)
print(resp1['answer'])

# 第二轮对话
resp2 = result_chain.invoke(
    {'input': 'What are common ways of doing it?'},
    config={'configurable': {'session_id': 'ls123456'}}
)
print(resp2['answer'])
