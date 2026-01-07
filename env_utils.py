import os
from dotenv import load_dotenv

# override 设置成 true 的作用：如果系统环境变量中存在同名的环境变量，.evn 文件中的文件将覆盖它
load_dotenv(override=True)

OPENAI_BASE_URL=os.getenv('OPENAI_BASE_URL')
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
# 未内置大模型：如 glm-4
GLM_OPENAI_BASE_URL=os.getenv('GLM_OPENAI_BASE_URL')
GLM_OPENAI_API_KEY=os.getenv('GLM_OPENAI_API_KEY')
# LangSmith
LANGSMITH_TRACING=os.getenv('LANGSMITH_TRACING')
LANGSMITH_ENDPOINT=os.getenv('LANGSMITH_ENDPOINT')
LANGSMITH_API_KEY=os.getenv('LANGSMITH_API_KEY')
LANGSMITH_PROJECT=os.getenv('LANGSMITH_PROJECT')
# Tavily
TAVILY_API_KEY=os.getenv('TAVILY_API_KEY')