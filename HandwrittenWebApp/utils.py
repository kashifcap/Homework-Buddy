from flask import send_file
import os


def save_from_file(file):
    filename = "text.txt"

    file.save(os.path.join(
        os.getcwd(), "HandwrittenModel/" + filename))


def save_from_text(text):
    filename = "text.txt"
    with open(os.path.join(os.getcwd(), "HandwrittenModel/" + filename), "w") as f:
        f.write(text)


def send_pdf_file():
    path = os.path.join(
        os.getcwd(), "HandwrittenModel/Handwritten.pdf")
    return send_file(path, as_attachment=True, attachment_filename="handwritten.pdf")
