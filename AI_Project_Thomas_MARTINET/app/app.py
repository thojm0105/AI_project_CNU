from flask import Flask, request, render_template, jsonify, session, redirect, url_for
from ultralytics import YOLO
import torch
from PIL import Image, ImageDraw
import pandas as pd
import os
import json
from datetime import datetime
import traceback

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Nécessaire pour utiliser les sessions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_011224.pt")
EXCEL_PATH = os.path.join(BASE_DIR, "Risk_assessments.xlsx")

# Charger le modèle YOLO
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit(1)

# Charger le fichier Excel
try:
    df_combined = pd.read_excel(EXCEL_PATH, sheet_name='Risques_Complexes')
    df_simple = pd.read_excel(EXCEL_PATH, sheet_name='Risques_Simples')
except Exception as e:
    print(f"Error loading Excel file: {e}")
    exit(1)

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"Unhandled exception: {traceback.format_exc()}")
    return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        first_name = request.form.get("first_name")
        last_name = request.form.get("last_name")
        user_id = request.form.get("user_id")
        manager_name = request.form.get("manager_name", "").strip()
        # On n'utilise plus is_manager pour déterminer le manager_name.
        # On prend toujours ce qui a été saisi.
        final_manager_name = manager_name if manager_name else "Not specified"

        session["user"] = {
            "first_name": first_name,
            "last_name": last_name,
            "user_id": user_id,
            "manager_name": final_manager_name
        }

        user_folder = os.path.join("static", "outputs", f"{first_name}_{last_name}_{user_id}")
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)

        return redirect(url_for("upload_page"))
    return render_template("index.html")


@app.route("/upload")
def upload_page():
    if "user" not in session:
        return redirect(url_for("home"))
    manager_name = session["user"].get("manager_name", "Not specified")
    return render_template("upload.html", manager_name=manager_name)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if "user" not in session:
            raise ValueError("User information missing.")

        if 'image' not in request.files:
            raise ValueError("No image file provided.")

        file = request.files['image']
        if not file.content_type.startswith('image/'):
            raise ValueError("Uploaded file is not an image.")

        image = Image.open(file.stream).convert("RGB")

        user = session["user"]
        user_folder = os.path.join("static", "outputs", f"{user['first_name']}_{user['last_name']}_{user['user_id']}")
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)

        with torch.no_grad():
            results = model(image)

        predictions = []
        detected_classes = []
        draw = ImageDraw.Draw(image)

        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            for box in results[0].boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                width = x_max - x_min
                height = y_max - y_min
                class_name = results[0].names[int(box.cls[0])]

                conf_value = float(box.conf[0])
                detected_classes.append(class_name)
                predictions.append({
                    "class": class_name,
                    "confidence": conf_value,
                    "bbox": [x_min, y_min, width, height]
                })

                display_confidence = f"{conf_value * 100:.2f}%"
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
                draw.text((x_min, y_min - 10), f"{class_name} {display_confidence}", fill="red")

        risks = get_risks(detected_classes)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_image_path = os.path.join(user_folder, f"annotated_image_{timestamp}.jpg")
        image.save(output_image_path)

        output_data_path = os.path.join(user_folder, f"predictions_{timestamp}.json")
        output_data = {
            "predictions": predictions,
            "risks": risks,
            "image_path": output_image_path
        }
        with open(output_data_path, 'w') as f:
            json.dump(output_data, f, indent=4)

        return jsonify(output_data)

    except Exception as e:
        print(f"Error in /predict: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

def get_risks(detected_classes):
    detected_set = set(detected_classes)
    combined_risks = []

    for _, row in df_combined.iterrows():
        combination = {row['Class 1'], row['Class 2']}
        if combination.issubset(detected_set):
            combined_risks.extend(row['Combined Risks'].split(", "))

    if combined_risks:
        return combined_risks

    individual_risks = []
    for cls in detected_classes:
        match = df_simple[df_simple['Class'] == cls]
        if not match.empty:
            individual_risks.append(match.iloc[0]['Risk'])

    return individual_risks if individual_risks else ["No specific risks identified"]

if __name__ == "__main__":
    if not os.path.exists("static/outputs"):
        os.makedirs("static/outputs")
    app.run(debug=True)
