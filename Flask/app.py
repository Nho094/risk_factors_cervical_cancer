from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Danh sách 35 đặc trưng đúng theo huấn luyện
feature_names = ['Age', 'Number of sexual partners', 'First sexual intercourse',
    'Num of pregnancies', 'Smokes', 'Smokes (years)', 'Smokes (packs/year)',
    'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD',
    'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis',
    'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
    'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
    'STDs:pelvic inflammatory disease', 'STDs:genital herpes',
    'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV',
    'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',
    'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis',
    'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller', 'Citology']

# Load model
model = joblib.load("logistic_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            values = [float(request.form[f]) for f in feature_names]
            prediction = model.predict([values])[0]
            proba = model.predict_proba([values])[0][1] * 100
            return render_template("index.html", features=feature_names,
                                   result=prediction, proba=round(proba, 2))
        except Exception as e:
            return f"Lỗi xử lý dữ liệu: {e}"
    return render_template("index.html", features=feature_names, result=None)

if __name__ == "__main__":
    app.run(debug=True)
