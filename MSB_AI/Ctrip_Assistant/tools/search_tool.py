from langchain_core.tools import BaseTool
from zai import ZhipuAiClient
from env_utils import GLM_OPENAI_API_KEY
from typing import Type
from pydantic import BaseModel, Field

client = ZhipuAiClient(api_key=GLM_OPENAI_API_KEY)

class SearchArgs(BaseModel):
    query: str = Field(description="需要进行联网搜索的信息")

class MySearchTool(BaseTool):
    name: str = "web_search"
    description: str = "搜索互联网上公开内容的工具"
    return_direct:bool = False # 搜索结果不直接返回，先给 agent
    args_schema:Type[BaseModel] = SearchArgs

    def _run(self, query: str) -> str:
        response = client.web_search.web_search(
            search_engine="search_pro",
            # search_engine="search_std",
            search_query=query
        )

        if response.search_result:
            return "\n\n".join([d.content for d in response.search_result])
        return '没有搜索到任何内容！'
