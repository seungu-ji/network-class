from flask import Flask, render_template, request
#from werkzeug import secure_filename

app = Flask(__name__)

@app.route("/")
def render_file():
    return render_template("image.html")

@app.route("/img_upload", methods=['POST'])
def upload_file():
    jsonData = request.get_json()
    print(jsonData)

    nameData = request.form['userEmail']
    print(nameData)

    f = request.files['filename']
    f.save('./img/' + f.filename)

    #radioData = request.form['image_chk']
    #print(radioData)

    return "hello world"
        

if __name__ == "__main__":
    app.run(debug = True)