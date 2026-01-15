from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model

from agent.my_llm import zhipuai_client


class SearchArgs(BaseModel):  # 类： 数据模型类
    query: str = Field(..., description='需要进行互联网查询的查询信息')
    # second: int = Field(..., description='第二个参数')


class MyWebSearchTool(BaseTool):
    name: str = "web_search2"  # 定义工具的名称

    description: str = "使用这个工具可以进行网络搜索。"
    # 第一种写法
    # args_schema: Type[BaseModel] = SearchArgs  # 工具的参数

    # 第二种写法
    def __init__(self):
        super().__init__()
        self.args_schema = create_model("SearchInput", query=(str, Field(..., description='需要进行互联网查询的查询信息')))

    def _run(self, query: str) -> str:
        try:
            # print("执行我的Python中的工具，输入的参数为:", query)
            response = zhipuai_client.web_search.web_search(
                search_engine="search_pro",
                search_query=query
            )
            # print(response)
            if response.search_result:
                return "\n\n".join([d.content for d in response.search_result])
            return '没有搜索到任何内容！'
        except Exception as e:
            print(e)
            return '没有搜索到任何内容！'

    async def _run(self, query: str) -> str:
        return self._run(query)