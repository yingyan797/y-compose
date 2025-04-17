from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=['get', 'post'])
def index():
    grid = []
    return render_template("index.html", backend_grid=grid)

@app.route("/save_grid", methods=['get', 'post'])
def save_grid():
    return render_template("index.html")


if __name__ == "__main__":
    app.run("0.0.0.0", 5002, debug=True)