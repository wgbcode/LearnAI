from Web import application
from wsgiref.simple_server import make_server

print("创建一个本地服务器")
httpd = make_server('', 8000, application)
print("开始监听请求，请在浏览器输入：http://127.0.0.1:8000/")
httpd.serve_forever()
