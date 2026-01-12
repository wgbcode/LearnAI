from test_mcp.mcp_server.mcp_tools import my_mcp_server

if __name__ == '__main__':
    my_mcp_server.run(
        transport="http",
        host="127.0.0.1",
        port=8000,
        path="/streamable",
        log_level="debug",
    )
