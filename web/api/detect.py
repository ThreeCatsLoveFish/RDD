from flask import render_template, request, jsonify
import subprocess


def detect():
    path = request.form['video_path']
    ps = subprocess.Popen(['python', 'inference.py', '--input', path], cwd='../model')
    ps.wait()
    return render_template('detect.html', video_path=path)
