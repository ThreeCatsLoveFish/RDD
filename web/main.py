from flask import Flask, render_template

from api import file, detect

app = Flask(__name__, template_folder='html', static_folder='static')


@app.route('/', methods=['GET'])
def Index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def UploadVideo():
    return file.upload()


@app.route('/detect', methods=['POST'])
def DetectVideo():
    return detect.detect()


if __name__ == '__main__':
    app.run('172.17.0.11', 9000)
