from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the home page"

@app.route('/test')
def test():
    return "RA ROI NE"

if __name__ == '__main__':
    app.run(debug=True)
