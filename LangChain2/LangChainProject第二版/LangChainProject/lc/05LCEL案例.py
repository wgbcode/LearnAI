from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

# 需求：  提示词1--> llm--> 文本----提示词2---->llm ---评分

llm = ChatOpenAI(
    temperature=0,
    model='deepseek-chat',
    api_key="sk-23308eef770c47e9aeb1149038ffb243",
    base_url="https://api.deepseek.com")

gather_preferences_prompt = ChatPromptTemplate.from_template(
    "用户输入了一些餐厅偏好：{input1}\n"
    "请将用户的偏好总结为清晰的需求："
)

recommend_restaurants_prompt = ChatPromptTemplate.from_template(
    "基于用户需求：{input2}\n"
    "请推荐 3 家适合的餐厅，并说明推荐理由："
)

# 步骤 3：总结推荐内容供用户快速参考
summarize_recommendations_prompt = ChatPromptTemplate.from_template(
    "以下是餐厅推荐和推荐理由：\n{input3}\n"
    "请总结成 2-3 句话，供用户快速参考："
)

chain = gather_preferences_prompt | llm | recommend_restaurants_prompt | llm | summarize_recommendations_prompt | llm | StrOutputParser()

print(chain.invoke({'input1': '我喜欢安静的地方， 有素食的餐厅更好，而且价格也不贵。'}))