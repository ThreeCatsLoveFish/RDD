from flask import render_template, request, jsonify
import subprocess
import os


def detect():
    path = request.form['video_path']
    file = os.path.join('..', 'web', path)
    ps = subprocess.Popen(['python', 'inference.py', '--input', file], cwd='../model', stdout=subprocess.PIPE)
    ps.wait()
    result = ps.stdout.readlines()
    for line in result:
        line = line.decode('utf-8')
        if line.startswith('Result'):
            detect =  line.split(';')[0].split(' ')[-1]
            confidence = line.split(';')[1].split(' ')[-1]

    return render_template('detect.html', detect=detect, confidence=confidence)
