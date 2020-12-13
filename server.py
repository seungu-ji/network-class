from flask import Flask, render_template, request
#from werkzeug import secure_filename
from cyclegan import *
from cartoongan import *
from timegan import *
import mysql.connector
import base64

mydb = mysql.connector.connect(
    host='database-2.cebw0vrctdih.us-east-1.rds.amazonaws.com',
    user='admin',
    port='3306',
    password='ksh03050621',
    database='Net'
)

mycursor = mydb.cursor()


app = Flask(__name__)

@app.route("/")
def render_file():
    return render_template("image.html")

@app.route("/img_upload", methods=['POST'])
def upload_file():
    print("응답함")

    nameData = request.form['userEmail']
    # print(nameData)

    f = request.files['filename']
    f.save('./img/' + f.filename)
    # print('./img/' + f.filename)
    path = './img/' + f.filename

    option = request.form.getlist('options')
    # print(option)

    if option[0] == '디즈니':
        cartoongan_predict(path)
    elif option[0] == '고흐':
        cyclegan_predict(path)
    else:
        print()
        print()
        print(option[0])
        timegan_predict(path, option[0])
    

    with open('./img/output.png', 'rb') as img:
        base64_string = base64.b64encode(img.read())

    # print(base64_string[len(base64_string)-100:])

    sql = "INSERT INTO user_image (userEmail, image) VALUES (%s, %s)"
    val = (nameData, base64_string)

    mycursor.execute(sql, val)

    mydb.commit()

    return "hello world"
        

if __name__ == "__main__":
    app.run(debug = True)