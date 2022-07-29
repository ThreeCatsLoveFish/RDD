from flask import render_template, request, jsonify
import uuid
import os


ALLOWED_EXTENSIONS = {'mp4'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def upload():
    f = request.files['file']
    if not (f and allowed_file(f.filename)):
        return jsonify({"error": 1001, "msg": "Please check the type of file, only mp4 is allowed!"})
    upload_path = os.path.join('static', 'videos', '{}.mp4'.format(uuid.uuid4()))
    print("Filename is %s, path is %s" % (f.filename, upload_path))
    f.save(upload_path)
    return render_template('video_preview.html', video_path=upload_path)
