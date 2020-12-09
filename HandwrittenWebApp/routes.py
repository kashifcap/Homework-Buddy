from HandwrittenWebApp import app
from HandwrittenModel import main
from HandwrittenWebApp.utils import save_from_file, save_from_text, send_pdf_file
from flask import render_template, request, url_for
import os


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                print('No file')
                return render_template('index.html')
            else:
                save_from_file(file)
                main()

                return send_pdf_file()

        elif 'text' in request.form:
            text = request.form['text']
            if text == '':
                print("No text")
                return render_template('index.html')
            else:
                save_from_text(text)

                main()

                return send_pdf_file()

    return render_template('index.html')
