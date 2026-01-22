"""
知识点：
一、创建 Langgraph-workflow_agent 项目
1、创建虚拟环境
（1）创建虚拟环境：python -m venv .venv
（2）激活虚拟环境：.\.venv\Scripts\activate
（3）退出虚拟环境：deactivate
（2）python 版本要大于 3.11
2、安装脚本 langgraph-cli
（1）pip install --upgrade "langgraph-cli[inmem]"
（2）inmem 是可选功能，英文是 in-memory，作用是在开发阶段增加内存存储的能力，而不用自己添加
3、创建项目模板
（1）不使用模板（交互式命令选择）：langgraph new
（2）使用模板：langgraph new my_langgraph --template new-langgraph-project-python
4、将 src 标记为源代码根目录（pycharm）

二、创建和使用 workflow_agent
（1）创建 workflow_agent 的方法又又又改变了，要通过 “from langchain.agents import create_agent” 导入使用
（2）传参方式要看官方文档

三、创建和使用 tool 工具
1、三种方法创建方法
（1）从函数创建工具（推荐）。@tools（本质是将函数包裹进了一个 wrapper 函数中）
（2）从可运行对象。StructuredTool.from_function
（3）子类化 BaseTool。as_tool
2、除了 description，也要记得给每个参数和返回值添加类型、作用提示，提高大模型的识别精度和输出准确率（也是三种方法）
（1）args_schema
（2）Annotated
（3）注释
3、注意事项
（1）工具的调用顺序是不确定的，由大模型决定
（2）所以，需要为每个工具提示清晰明确的 description
（3）在一个工具执行完毕后，可以要求它优先执行哪一个工具

四、添加上下文功能
1、Configurable（由用户传进来）
2、AgentState
（1）类似于 Vue3 中的 Pinia，是一个全局数据管理库，支持读写每一轮的对话记录
（2）如果不传，会有一个默认的 State
（3）如果传，可以添加自定义的字段，并且可以通过 Command 方法，来修改每一轮的 State

五、添加网络搜索功能
1、通过添加一个工具实现
2、需要使用外部的大模型，如智谱 ZhipuAI

六、添加短期记忆功能（对话）、长期记忆功能（会话）
1、短期记忆配置字段：checkpointer
2、长期记忆配置字段：store（基于短期记忆实现）

七、使用 MCP 调用工具
1、MCP 协议（基于 http 和 sse，sse 又基于 websocket）
（1）同时支持 sse 和 post
（2）同时支持有状态和无状态服务器（无状态服务器无法使用 sse 建立长连接）
（3）调 MCP 工具时，要异步，即 workflow_agent.ainvoke
2、三种创建/调用 MCP 服务的方式
（1）FastMCP (基于 fastapi 实现）
（2）java（SpringAI）
（3）远程 MCP 服务
3、权限认证和安全：token
（1）后端创建 token
（2）后端配置 token
（3）前端传 token

八、工作流 WorkFlow
1、节点 node（函数；agent）
2、边 edge（包含路由 router）
3、状态 state（基类为 MessageState）

九、智能小秘书案例【从0到1实现】
1、三个外部 MCP（魔塔；百炼）
（1）可以帮我查询火车票（火车票查询 MCP）
（2）可以帮我生成分析图表（图表 MCP）
（3）可以帮我获取网络上的数据（普通数据查询 MCP）
2、核心知识点（结合工作流实现）
（1）异步
（2）并发
（3）参数验证
（4）错误处理
（5）人工介入(Human-in-the-loop。llm => human => tools；验证/修改状态/增加一些新的值；中断工作流）
（6）流式输出
3、四种消息类型
（1）HumanMessage
（2）AIMessage
（3）ToolsMessage
（4）SystemMessage

十、Text-To-SQL
"""