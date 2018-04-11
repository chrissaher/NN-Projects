# from style_transfer import StyleTransfer

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

@app.route('/api/nst', methods=['POST'])
def nst():
    try:
        name = request.json
        lastname = name
        print("name: ", name)
        print("Lastname: ", lastname)
        msg = "hello " + str(name) + " " + str(lastname)
        print('Msg: ', msg)
    except ex:
        print(ex)
    return jsonify(msg=msg)

@app.route('/')
def main():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
