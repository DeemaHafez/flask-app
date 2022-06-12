"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
#from asyncio.windows_events import NULL
import io
import os
from PIL import Image
import glob
import torch
from flask import Flask, render_template, request, redirect
import json
import cv2
import numpy as np
import easyocr
from flask import jsonify
import re
reader = easyocr.Reader(['ar'])

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model(img, size=640)
        crops = results.crop(save=True)

        # Arabic OCR
        dir_path = os.path.abspath(os.getcwd())
        dir_path = dir_path.replace("\\", "/") + "/"
        detection_folder = glob.glob('runs/detect/*')
        latest_folder = max(detection_folder, key=os.path.getctime)
        local_download_path = latest_folder + "/crops"
        crops = glob.glob(local_download_path + '/*')
        info = {}
        for fold in crops:
            class_folder_name = os.path.basename(fold)
            if class_folder_name == "full_name":
                if 1 < len(glob.glob(local_download_path + '/full_name/*')):
                    image_path = glob.glob(
                        local_download_path + '/full_name/*')[1]
                else:
                    image_path = glob.glob(
                        local_download_path + '/full_name/*')[0]
                image_path = image_path.replace("\\", "/")
                ig = cv2.imread(dir_path+image_path)
                bounds = reader.readtext(ig)
                word = ""
                for bound in bounds:
                    if not re.search(r"\d", bound[1]):
                        if len(bound[1]) > 2:
                            word += ' ' + bound[1]
                info['full_name'] = word

            if(class_folder_name == "id"):
                image_path = glob.glob(local_download_path + '/id/*')[0]
                image_path = image_path.replace("\\", "/")
                ig = cv2.imread(dir_path+image_path)
                bounds = reader.readtext(ig)
                word = ""
                for bound in bounds:
                    word = bound[1]
                info['id'] = word.strip()

            if(class_folder_name == "birth_date"):
                image_path = glob.glob(
                    local_download_path + '/birth_date/*')[0]
                image_path = image_path.replace("\\", "/")
                ig = cv2.imread(dir_path+image_path)
                bounds = reader.readtext(ig)
                word = ""
                for bound in bounds:
                    word = bound[1]
                info['birth_date'] = word.strip()

            if(class_folder_name == "birth_place"):
                image_path = glob.glob(
                    local_download_path + '/birth_place/*')[0]
                image_path = image_path.replace("\\", "/")
                ig = cv2.imread(dir_path+image_path)
                bounds = reader.readtext(ig)
                word = ""
                for bound in bounds:
                    word += ' ' + bound[1]
                info['birth_place'] = word.strip()

            if(class_folder_name == "first_name"):
                if 1 < len(glob.glob(local_download_path + '/first_name/*')):
                    image_path = glob.glob(
                        local_download_path + '/first_name/*')[1]
                else:
                    image_path = glob.glob(
                        local_download_path + '/first_name/*')[0]
                image_path = image_path.replace("\\", "/")
                ig = cv2.imread(dir_path+image_path)
                bounds = reader.readtext(ig)
                word = ""
                for bound in bounds:
                    word += ' ' + bound[1]
                info['first_name'] = word.strip()

            if(class_folder_name == "second_name"):
                if 1 < len(glob.glob(local_download_path + '/second_name/*')):
                    image_path = glob.glob(
                        local_download_path + '/second_name/*')[1]
                else:
                    image_path = glob.glob(
                        local_download_path + '/second_name/*')[0]
                image_path = image_path.replace("\\", "/")
                ig = cv2.imread(dir_path+image_path)
                bounds = reader.readtext(ig)
                word = ""
                for bound in bounds:
                    word += ' ' + bound[1]
                info['second_name'] = word.strip()

            if(class_folder_name == "grand_name"):
                if 1 < len(glob.glob(local_download_path + '/grand_name/*')):
                    image_path = glob.glob(
                        local_download_path + '/grand_name/*')[1]
                else:
                    image_path = glob.glob(
                        local_download_path + '/grand_name/*')[0]
                image_path = image_path.replace("\\", "/")
                ig = cv2.imread(dir_path+image_path)
                bounds = reader.readtext(ig)
                word = ""
                for bound in bounds:
                    word += ' ' + bound[1]
                info['grand_name'] = word.strip()

            if(class_folder_name == "family_name"):
                if 1 < len(glob.glob(local_download_path + '/family_name/*')):
                    image_path = glob.glob(
                        local_download_path + '/family_name/*')[1]
                else:
                    image_path = glob.glob(
                        local_download_path + '/family_name/*')[0]
                image_path = image_path.replace("\\", "/")
                ig = cv2.imread(dir_path+image_path)
                bounds = reader.readtext(ig)
                word = ""
                for bound in bounds:
                    word += ' ' + bound[1]
                info['family_name'] = word.strip()
        print(info)
        return jsonify(info)

    return render_template("index.html")


if __name__ == "__main__":
    dir_path = os.path.abspath(os.getcwd())
    # parser = argparse.ArgumentParser(
    #     description="Flask app exposing yolov5 models")
    # parser.add_argument("--port", default=5000, type=int, help="port number")
    # args = parser.parse_args()

    model = torch.hub.load(
        'ultralytics/yolov5', 'custom', path=dir_path+"/best2.pt"
    )

    model.eval()
    # debug=True causes Restarting with stat
    #PORT = int(os.environ.get('PORT', 5000))
    app.run(debug=True)
