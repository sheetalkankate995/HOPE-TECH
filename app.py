
from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, session
)
from functools import wraps
import os
import json
import joblib
import sys

from database import (
    initialize_database,
    verify_user,
    register_user
)

app = Flask(__name__)

# Session security key

app.secret_key = os.environ.get("SECRET_KEY")
# Initialize database
initialize_database()

# Project folder
PROJECT_FOLDER = r"C:\Project by us\SuicidalIdeationProject"
MODEL_FOLDER = os.path.join(PROJECT_FOLDER, "models")

# --------------------------------
# LOAD ML MODELS
# --------------------------------

# Original ML model
model = joblib.load(
    os.path.join(MODEL_FOLDER, "model.pkl")
)

vectorizer = joblib.load(
    os.path.join(MODEL_FOLDER, "vectorizer.pkl")
)

# Genetic Algorithm model
ga_model = joblib.load(
    os.path.join(MODEL_FOLDER, "ga_model.pkl")
)

ga_vectorizer = joblib.load(
    os.path.join(MODEL_FOLDER, "ga_vectorizer.pkl")
)

selected_features = joblib.load(
    os.path.join(MODEL_FOLDER, "selected_features.pkl")
)

# --------------------------------
# LOAD ACTUAL MODEL METRICS
# --------------------------------

def load_model_metrics(filename, display_name):

    metrics_path = os.path.join(MODEL_FOLDER, filename)

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}"
        )

    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    # Preserve the existing dashboard API field names
    metrics["name"] = display_name

    return metrics


model_results = {
    "original": load_model_metrics(
        "model_metrics.json",
        "Original ML Model"
    ),

    "ga": load_model_metrics(
        "ga_model_metrics.json",
        "Genetic Algorithm Model"
    )
}

# --------------------------------
# IMPORT PREPROCESSING
# --------------------------------

sys.path.append(os.path.join(PROJECT_FOLDER, "src"))

from preprocessing import clean_text

# --------------------------------
# LIVE PREDICTION COUNTERS
# --------------------------------

prediction_counts = {
    "original": 0,
    "ga": 0
}


# --------------------------------
# LOGIN PROTECTION
# --------------------------------

def login_required(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        if "user_id" not in session:
            return redirect(url_for("home"))

        return function(*args, **kwargs)

    return wrapper


def role_required(*allowed_roles):

    def decorator(function):

        @wraps(function)
        def wrapper(*args, **kwargs):

            if "user_id" not in session:
                return jsonify({
                    "error": "Please log in first."
                }), 401

            if session.get("role") not in allowed_roles:
                return jsonify({
                    "error": "Access denied."
                }), 403

            return function(*args, **kwargs)

        return wrapper

    return decorator


# --------------------------------
# ABOUT US PAGE - HOME PAGE
# --------------------------------

@app.route('/')
def home():

    if "user_id" in session:
        return redirect(url_for("dashboard"))

    return render_template("about.html")


@app.route('/about')
def about():

    return render_template("about.html")


# --------------------------------
# REGISTRATION PAGE
# --------------------------------

@app.route('/register', methods=['GET', 'POST'])
def register():

    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == 'GET':
        return render_template("register.html")

    username = request.form.get("username", "").strip()
    mobile_number = request.form.get("mobile_number", "").strip()
    full_name = request.form.get("full_name", "").strip()
    age_text = request.form.get("age", "").strip()
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")
    role = request.form.get("role", "").strip().lower()

    if not all([
        username,
        mobile_number,
        full_name,
        age_text,
        email,
        password,
        confirm_password,
        role
    ]):

        return render_template(
            "register.html",
            error="Please fill in all fields."
        )

    if not age_text.isdigit():

        return render_template(
            "register.html",
            error="Age must be a valid number."
        )

    age = int(age_text)

    if age < 1 or age > 120:

        return render_template(
            "register.html",
            error="Please enter a valid age."
        )

    if password != confirm_password:

        return render_template(
            "register.html",
            error="Passwords do not match."
        )

    if role not in ("student", "faculty"):

        return render_template(
            "register.html",
            error="Please select Student or Faculty."
        )

    success, message = register_user(
        username,
        mobile_number,
        full_name,
        age,
        email,
        password,
        role
    )

    if not success:

        return render_template(
            "register.html",
            error=message
        )

    return render_template(
        "register.html",
        success=message
    )


# --------------------------------
# LOGIN PAGE
# --------------------------------

@app.route('/login', methods=['GET', 'POST'])
def login():

    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == 'GET':
        return render_template("login.html")

    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")
    role = request.form.get("role", "").strip().lower()

    if not email or not password or role not in (
        "admin", "faculty", "student"
    ):

        return render_template(
            "login.html",
            error="Please select a role and enter your login details."
        )

    user = verify_user(email, password, role)

    if user is None:

        return render_template(
            "login.html",
            error="Invalid email, password, or selected role."
        )

    session.clear()

    session["user_id"] = user["id"]
    session["full_name"] = user["full_name"]
    session["email"] = user["email"]
    session["role"] = user["role"]

    return redirect(url_for("dashboard"))


# --------------------------------
# DASHBOARD
# --------------------------------

@app.route('/dashboard')
@login_required
def dashboard():

    return render_template(
        "dashboard.html",
        user_name=session.get("full_name"),
        user_role=session.get("role")
    )


# --------------------------------
# LOGOUT
# --------------------------------

@app.route('/logout')
@login_required
def logout():

    session.clear()

    return redirect(url_for("home"))


# --------------------------------
# ML PREDICTION
# Admin and Student access
# --------------------------------

@app.route('/predict', methods=['POST'])
@role_required("admin", "student")
def predict():

    try:

        data = request.get_json(silent=True) or {}
        text = data.get("text", "").strip()

        if not text:
            return jsonify({
                "error": "Please enter some text."
            }), 400

        cleaned_text = clean_text(text)

        # Original ML prediction
        original_data = vectorizer.transform([cleaned_text])

        original_prediction = model.predict(original_data)[0]

        original_probabilities = model.predict_proba(
            original_data
        )[0]

        original_classes = model.classes_

        original_probability = {
            str(label): round(float(prob), 4)
            for label, prob in zip(
                original_classes,
                original_probabilities
            )
        }

        # Genetic Algorithm prediction
        ga_data = ga_vectorizer.transform([cleaned_text])

        ga_data = ga_data[:, selected_features]

        ga_prediction = ga_model.predict(ga_data)[0]

        ga_probabilities = ga_model.predict_proba(ga_data)[0]

        ga_classes = ga_model.classes_

        ga_probability = {
            str(label): round(float(prob), 4)
            for label, prob in zip(
                ga_classes,
                ga_probabilities
            )
        }

        # Update prediction counters
        prediction_counts["original"] += 1
        prediction_counts["ga"] += 1

        return jsonify({

            "original_prediction": str(original_prediction),

            "ga_prediction": str(ga_prediction),

            "original_probabilities": original_probability,

            "ga_probabilities": ga_probability,

            "prediction_counts": prediction_counts

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------
# LIVE ANALYTICS
# Admin and Faculty access
# --------------------------------

@app.route('/api/analytics')
@role_required("admin", "faculty")
def analytics():

    return jsonify({
        "prediction_counts": prediction_counts
    })


# --------------------------------
# ACTUAL MODEL EVALUATION RESULTS
# Admin and Faculty access
# --------------------------------

@app.route('/api/model-results')
@role_required("admin", "faculty")
def model_results_api():

    return jsonify(model_results)


# --------------------------------
# RUN FLASK APPLICATION
# --------------------------------

if __name__ == '__main__':

    app.run(
        debug=True,
        use_reloader=False
    )
