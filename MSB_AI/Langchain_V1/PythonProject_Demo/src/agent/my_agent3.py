from typing import List, Optional

from langchain.agents import create_agent, AgentState
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from agent.my_llm import llm
from agent.toots.test_to_sql_tools import ListTablesTool, TableSchemaTool, SQLQueryTool, SQLQueryCheckerTool
from agent.utils.db_utils import MySQLDatabaseManager


def get_tools(host: str, port: int, username: str, password: str, database: str) -> List[BaseTool]:
    # 构建连接字符串
    connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}?charset=utf8mb4"
    manager = MySQLDatabaseManager(connection_string)
    return [
        ListTablesTool(db_manager=manager),
        TableSchemaTool(db_manager=manager),
        SQLQueryTool(db_manager=manager),
        SQLQueryCheckerTool(db_manager=manager),
    ]


tools = get_tools('127.0.0.1', 3306, 'root', '123123', 'test_db4')

system_prompt = """
你是一个专门设计用于与SQL数据库交互的AI智能体。

给定一个输入问题，你需要按照以下步骤操作：
1. 创建一个语法正确的{dialect}查询语句
2. 执行查询并查看结果
3. 基于查询结果返回最终答案

除非用户明确指定要获取的具体示例数量，否则始终将查询结果限制为最多{top_k}条。

你可以通过相关列对结果进行排序，以返回数据库中最有意义的示例。
永远不要查询特定表的所有列，只获取与问题相关的列。

在执行查询之前，你必须仔细检查查询语句。如果在执行查询时遇到错误，请重写查询并再次尝试。

绝对不要对数据库执行任何数据操作语言（DML）语句（如INSERT、UPDATE、DELETE、DROP等）。

开始处理问题时，你应该始终先查看数据库中有哪些表可以查询。不要跳过这一步。

然后，你应该查询最相关表的模式结构信息。
""".format(
    dialect='MySQL',  # 数据库方言（如MySQL、SQLite等）
    top_k=5,  # 默认返回结果的最大数量
)




#  自定义 AgentState，或者Context
class CustomAgentState(AgentState):
    user_name: Optional[str]  # 我自己 增加的， 没有提供reducer函数则覆盖

class CustomContext(BaseModel):
    """自定义的静态状态"""
    user_id: Optional[str]  # 我自己 增加的，

agent = create_agent(  # 默认有AgentState 参数, messages的key，一个消息列表
    llm,
    tools=tools,
    system_prompt=system_prompt,
    state_schema=CustomAgentState,
    context_schema=CustomContext,
)

if __name__ == '__main__':
    # system_msg = SystemMessage("You are a helpful coding assistant.")
    # messages = [
    #     system_msg,
    #     HumanMessage("How do I create a REST API?")
    # ]
    for step in agent.stream(
            input={'messages': [{'role': 'user', 'content': '数据库中有多少个部门，每个部门都有哪些员工？'}]},
            context={'user_id': '123'},
            # context=CustomContext(user_id='123'),,
            stream_mode="values",
    ):
        step['messages'][-1].pretty_print()  # 打印
