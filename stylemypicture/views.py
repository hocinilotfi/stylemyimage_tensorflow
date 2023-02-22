#import numpy as np
#import cv2
import base64
import gc
import io
import os
from fileinput import filename

from flask import (Flask, flash, redirect, render_template, request, send_file,
                   url_for)
from PIL import Image
from stylemypicture import app
from stylemypicture.processor import *
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'jpg', 'png', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def PIL_to_HTML_display(PIL_img):
    '''
    using this function to avoid saving data on disk / for realtime  processing
    in python:
        return render_template("index.html", data=PIL_to_HTML_display(PIL_img))
    in html:
        <img src="data:image/jpeg;base64,{{data }}" alt="" width="480px" height="360px">
    '''
    binary_buffer = io.BytesIO()
    PIL_img.save(binary_buffer, "JPEG")
    encoded_img = base64.b64encode(binary_buffer.getvalue())
    return encoded_img.decode('utf-8')


style_image_list = ["im01", "im02", "im03", "im04", "im05", "im06", "im07", "im08",
                    "im09", "im10", "im11", "im12", "im13", "im14", "im15",
                    "im16", "im17", "im18", "im19", "im20", "im21", "im22", "im23"]




@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            '''
                Convert image grom PIL to opencv
                I = numpy.asarray(PIL.Image.open('test.jpg'))
                Do some stuff to I, then, convert it back to an image:
                im = PIL.Image.fromarray(numpy.uint8(I))
            '''
            # read image from request without saving it on disk
            image = request.files['file'].read()
            image = io.BytesIO(image)

            image = Image.open(image).convert('RGB')

            img_style_number = request.form['slidenumber']

            path_to_choosed_style = os.path.join('stylemypicture', 'static', 'images',
                                                 str(style_image_list[int(img_style_number)]) + ".jpg")
            img = process_my_picture( path_to_choosed_style, image)

            data = {
                # "out": path,

                "processed_img": PIL_to_HTML_display(img)
                # "received_img": PIL_to_HTML_display(Image.open(image)),

            }
            return render_template("index.html", data=data)
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')
