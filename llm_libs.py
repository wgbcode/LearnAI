from env_utils import OPENAI_BASE_URL,OPENAI_API_KEY,GLM_OPENAI_BASE_URL,GLM_OPENAI_API_KEY,QWEN_OPENAI_BASE_URL,QWEN_OPENAI_API_KEY
from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(
#     model="qwen3-8b",
#     base_url=QWEN_OPENAI_BASE_URL,
#     api_key=QWEN_OPENAI_API_KEY
# )

llm = ChatOpenAI(
    model="gpt-5-mini",
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY
)

# llm = ChatOpenAI(
#     model="glm-4",
#     base_url=GLM_OPENAI_BASE_URL,
#     api_key=GLM_OPENAI_API_KEY
# )