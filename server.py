from flask import Flask, render_template, request
#from werkzeug import secure_filename
from cyclegan import *
import mysql.connector

mydb = mysql.connector.connect(
    host='10.53.68.126',
    user='root',
    port='3306',
    password='password123@',
    database='Net'
)

mycursor = mydb.cursor()


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
    
    cyclegan_binary = cyclegan_predict(path)
    print(cyclegan_binary)
    
    sql = "INSERT INTO user_image (userEmail, imgage) VALUES (%s, %s)"
    val = (nameData, cyclegan_binary)

    mycursor.execute(sql, val)

    mydb.commit()

    #radioData = request.form['image_chk']
    #print(radioData)

    return "hello world"
        

if __name__ == "__main__":
    app.run(debug = True)