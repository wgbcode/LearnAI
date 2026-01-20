from flask import Flask, request,render_template

"""
其它 web 开发框架：
1、Djanggo: 全能型 Web 框架
2、web.py：一个小巧的 web 框架
3、Bottle：和 Flask 类似
4、Tornado：Facebook 的开源异步 web 框架
5、FastAPI：vue3+，只构建后端 API 接口，能自动生成接口文档（Flask：简单API）
"""

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def hello_world():
    # return "<p>Hello, World!</p>"
    return render_template('index.html')

@app.route('/signin', methods=['GET'])
def signin_form():
    return '''<form action="/signin" method="post">
              <p><input name="username"></p>
              <p><input name="password" type="password"></p>
              <p><button type="submit">Sign In</button></p>
              </form>'''

@app.route('/signin', methods=['POST'])
def signin():
    # 需要从request对象读取表单内容：
    if request.form['username']=='admin' and request.form['password']=='password':
        return '<h3>Hello, admin!</h3>'
    return '<h3>Bad username or password.</h3>'

if __name__ == '__main__':
    app.run()

