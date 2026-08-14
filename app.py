# flask_api/app.py

from flask import Flask

app = Flask(__name__)


@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "service": "ContractGuardian AI"
    }


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )