from flask import Flask, render_template, request
#from werkzeug import secure_filename
from cyclegan import *

app = Flask(__name__)

@app.route("/")
def render_file():
    return render_template("image.html")

@app.route("/img_upload", methods=['POST'])
def upload_file():
    """jsonData = request.get_json()
    print(jsonData)"""

    nameData = request.form['userEmail']
    # print(nameData)

    f = request.files['filename']
    f.save('./img/' + f.filename)
    # print('./img/' + f.filename)
    path = './img/' + f.filename

    # print(path.type)
    #select = 0

    # if select == "cyclegan":
    print(cyclegan_predict(path))
    

    #radioData = request.form['image_chk']
    #print(radioData)

    return "hello world"
        

if __name__ == "__main__":
    app.run(debug = True)