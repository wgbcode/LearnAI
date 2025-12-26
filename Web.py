def application(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/html')])
    method = environ['REQUEST_METHOD']
    path = environ['PATH_INFO']
    print(f"请求方法：{method},请求路径：{path}")
    return [b'<h1>Hello World!</h1>']
