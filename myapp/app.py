from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import datetime
import numpy as np
import pandas as pd
import joblib
import os
import json
from dotenv import load_dotenv
import threading
import requests
import shap 
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Lock thread-safe
save_lock = threading.Lock()

def save_history_to_file(record):
    filename = "history_data.json"
    with save_lock:
        if os.path.exists(filename):
            with open(filename, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        data.append(record)
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

def ask_openrouter(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "CervicalCancerPredictor"
    }

    payload = {
        "model": "google/gemma-3n-e2b-it:free",  # hoặc "mistralai/mixtral-8x7b" nếu bạn muốn
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ Lỗi API ({response.status_code}): {response.text}"
    except requests.exceptions.Timeout:
        return "❌ Lỗi: Quá thời gian phản hồi từ OpenRouter."
    except Exception as e:
        return f"❌ Lỗi: {str(e)}"


from feature_advice import feature_advice

app = Flask(__name__)
app.secret_key = "super-secret-key"

feature_names = ['Age', 'Number of sexual partners', 'First sexual intercourse',
    'Num of pregnancies', 'Smokes', 'Smokes (years)', 'Smokes (packs/year)',
    'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD',
    'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis',
    'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
    'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
    'STDs:pelvic inflammatory disease', 'STDs:genital herpes',
    'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV',
    'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',
    'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller', 'Citology']

label_mapping = {
    "Age": "Tuổi", "Number of sexual partners": "Số bạn tình", "First sexual intercourse": "Tuổi quan hệ lần đầu",
    "Num of pregnancies": "Số lần mang thai", "Smokes": "Hút thuốc", "Smokes (years)": "Số năm hút thuốc",
    "Smokes (packs/year)": "Số gói mỗi năm", "Hormonal Contraceptives": "Dùng thuốc tránh thai",
    "Hormonal Contraceptives (years)": "Số năm dùng thuốc tránh thai", "IUD": "Đặt vòng",
    "IUD (years)": "Số năm đặt vòng", "STDs": "Từng mắc STDs", "STDs (number)": "Số lần mắc STDs",
    "STDs:condylomatosis": "Sùi mào gà", "STDs:cervical condylomatosis": "Sùi mào gà cổ tử cung",
    "STDs:vaginal condylomatosis": "Sùi mào gà âm đạo", "STDs:vulvo-perineal condylomatosis": "Sùi mào gà âm hộ - tầng sinh môn",
    "STDs:syphilis": "Bệnh giang mai", "STDs:pelvic inflammatory disease": "Viêm vùng chậu",
    "STDs:genital herpes": "Mụn rộp sinh dục", "STDs:molluscum contagiosum": "U mềm lây",
    "STDs:AIDS": "AIDS", "STDs:HIV": "HIV", "STDs:Hepatitis B": "Viêm gan B", "STDs:HPV": "Nhiễm HPV",
    "STDs: Number of diagnosis": "Số lần được chẩn đoán STDs", "Dx:Cancer": "Chẩn đoán ung thư",
    "Dx:CIN": "Chẩn đoán CIN", "Dx:HPV": "Chẩn đoán HPV", "Dx": "Chẩn đoán bất thường",
    "Hinselmann": "Hinselmann dương tính", "Schiller": "Schiller dương tính", "Citology": "Tế bào học dương tính"
}

model = joblib.load("random_forest_model.pkl")
print("✅ Đã load mô hình:", type(model))
explainer = shap.TreeExplainer(model)  # đặt ở ngoài
# def generate_advice_simple(proba):
#     if proba >= 25:
#         return (
#             "🔴 CẢNH BÁO: Có nguy cơ tiềm ẩn.\n\n"
#             "💡 Khuyến nghị:\n"
#             "• Tham khảo ý kiến bác sĩ chuyên khoa\n"
#             "• Tiến hành xét nghiệm Pap smear hoặc HPV nếu chưa làm\n"
#             "• Duy trì lối sống lành mạnh\n"
#             "• Tránh thuốc lá, hạn chế rượu bia\n"
#             "• Tiêm vaccine HPV nếu chưa tiêm\n"
#             "❤️ Sức khỏe của bạn là điều quan trọng nhất."
#         )
#     else:
#         return (
#             "✅ Bạn hiện không có nguy cơ đáng kể.\n\n"
#             "💡 Gợi ý:\n"
#             "• Khám phụ khoa định kỳ\n"
#             "• Duy trì lối sống lành mạnh\n"
#             "• Tránh hút thuốc, hạn chế rượu bia\n"
#             "• Tham khảo tiêm vaccine HPV nếu chưa tiêm\n"
#             "❤️ Chúc bạn luôn khoẻ mạnh."
#         )

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         try:
#             input_dict = dict.fromkeys(feature_names, np.nan)
#             for name in feature_names:
#                 raw_val = request.form.get(name)
#                 try:
#                     input_dict[name] = float(raw_val)
#                 except (TypeError, ValueError):
#                     pass

#             # Nếu tất cả đều rỗng
#             if pd.DataFrame([input_dict]).isna().all(axis=1).values[0]:
#                 return "⚠️ Vui lòng nhập ít nhất một giá trị vào biểu mẫu."

#             X_input = pd.DataFrame([input_dict], columns=feature_names).fillna(0)
#             X_input = X_input[model.feature_names_in_]

#             prediction = model.predict(X_input)[0]
#             proba = model.predict_proba(X_input)[0][1] * 100
#             advice = generate_advice_simple(proba)

#             # ✅ Tạo extra_insight từ input và feature_advice
#             extra_insight = ""
#             for name in feature_names:
#                 val = input_dict[name]
#                 if val and name in feature_advice:
#                     vi_name = label_mapping.get(name, name)
#                     desc = feature_advice[name].get("desc", "")
#                     action = feature_advice[name].get("action", "")
#                     extra_insight += f"• {vi_name} = {val}\n  {desc}\n  👉 {action}\n\n"

#             record = {
#                 "input": {k: request.form.get(k) for k in feature_names},
#                 "result": int(prediction),
#                 "proba": round(proba, 2),
#                 "advice": advice,
#                 "timestamp": datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
#             }

#             if "history" not in session:
#                 session["history"] = []
#             session["history"].append(record)
#             session.modified = True
#             save_history_to_file(record)

#             return render_template("index.html", features=feature_names,
#                                    result=prediction, proba=round(proba, 2),
#                                    advice=advice, extra_insight=extra_insight)
#         except Exception as e:
#             return f"Lỗi xử lý dữ liệu: {e}"
#     return render_template("index.html", features=feature_names, result=None)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            input_dict = dict.fromkeys(feature_names, np.nan)
            for name in feature_names:
                raw_val = request.form.get(name)
                try:
                    input_dict[name] = float(raw_val)
                except (TypeError, ValueError):
                    pass

            if pd.DataFrame([input_dict]).isna().all(axis=1).values[0]:
                return "⚠️ Vui lòng nhập ít nhất một giá trị vào biểu mẫu."

            # Dự đoán
            X_input = pd.DataFrame([input_dict], columns=feature_names).fillna(0)
            X_input = X_input[model.feature_names_in_]
            prediction = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0][1] * 100

            # Sinh lời khuyên chính
            if proba >= 25:
                advice = (
                    "🔴 CẢNH BÁO: Có nguy cơ tiềm ẩn.\n\n"
                    "💡 Khuyến nghị:\n"
                    "• Tham khảo ý kiến bác sĩ chuyên khoa\n"
                    "• Tiến hành xét nghiệm Pap smear hoặc HPV nếu chưa làm\n"
                    "• Duy trì lối sống lành mạnh\n"
                    "• Tránh thuốc lá, hạn chế rượu bia\n"
                    "• Tiêm vaccine HPV nếu chưa tiêm\n"
                    "❤️ Sức khỏe của bạn là điều quan trọng nhất."
                )
            else:
                advice = (
                    "✅ Bạn hiện không có nguy cơ đáng kể.\n\n"
                    "💡 Gợi ý:\n"
                    "• Khám phụ khoa định kỳ\n"
                    "• Duy trì lối sống lành mạnh\n"
                    "• Tránh hút thuốc, hạn chế rượu bia\n"
                    "• Tham khảo tiêm vaccine HPV nếu chưa tiêm\n"
                    "❤️ Chúc bạn luôn khoẻ mạnh."
                )

            # Phân tích từ các đặc trưng nguy cơ (dựa trên threshold)
            extra_insight = ""
            for name in feature_names:
                val = input_dict[name]
                if pd.notna(val) and name in feature_advice:
                    threshold = feature_advice[name].get("threshold", 0)
                    if val >= threshold:
                        vi_name = label_mapping.get(name, name)
                        desc = feature_advice[name].get("desc", "")
                        action = feature_advice[name].get("action", "")
                        extra_insight += f"• {vi_name} = {val}\n  {desc}\n  👉 {action}\n\n"
                    elif "desc_low" in feature_advice[name]:
                        vi_name = label_mapping.get(name, name)
                        desc = feature_advice[name].get("desc_low", "")
                        action = feature_advice[name].get("action_low", "")
                        extra_insight += f"• {vi_name} = {val}\n  {desc}\n  👉 {action}\n\n"

            # Phân tích bổ sung từ OpenRouter (nếu cần)
            try:
                extra_ai = ask_openrouter(f"Hãy phân tích lời khuyên y khoa sau bằng tiếng Việt:\n{advice}")
            except:
                extra_ai = ""

            # Lưu lịch sử
            record = {
                "input": {k: request.form.get(k) for k in feature_names},
                "result": int(prediction),
                "proba": round(proba, 2),
                "advice": advice,
                "timestamp": datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
            }

            if "history" not in session:
                session["history"] = []
            session["history"].append(record)
            session.modified = True
            save_history_to_file(record)

            return render_template("index.html",
                                   features=feature_names,
                                   result=prediction,
                                   proba=round(proba, 2),
                                   advice=advice,
                                   extra_insight=extra_insight + extra_ai)
        except Exception as e:
            return f"Lỗi xử lý dữ liệu: {e}"

    return render_template("index.html", features=feature_names, result=None)


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        if not prompt.strip():
            return jsonify({"reply": "⚠️ Không nhận được câu hỏi hợp lệ."})
        reply = ask_openrouter(prompt)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Lỗi: {e}"})


@app.route("/monitor")
def monitor():
    try:
        with open("history_data.json", "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []

    # Lấy nhãn thời gian và xác suất tương ứng
    labels = [entry.get("timestamp", f"Lần {i+1}") for i, entry in enumerate(history)]
    probabilities = [entry.get("proba", 0) for entry in history]

    return render_template("monitor.html",
                           labels=labels,
                           probabilities=probabilities,
                           history=history)



@app.route("/predict", methods=["POST"])
def predict():
    return redirect(url_for("index"))

@app.route('/history')
def history():
    try:
        with open("history_data.json", "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []
    return render_template("history.html", history=history, label_mapping=label_mapping)

@app.route("/clear_history")
def clear_history():
    session.pop("history", None)
    if os.path.exists("history_data.json"):
        os.remove("history_data.json")
    return redirect(url_for("monitor"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
