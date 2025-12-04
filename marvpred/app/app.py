import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SubmitField
from wtforms.validators import DataRequired
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib

# App Configuration
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "a_hard_to_guess_string")


# --- Forms ---
class PredictionForm(FlaskForm):
    """Form for SMILES input and model selection."""

    smiles = StringField(
        "Canonical SMILES",
        validators=[DataRequired()],
        render_kw={"placeholder": "e.g., CCO"},
    )
    model = SelectField("Select Model", choices=[], validators=[DataRequired()])
    submit = SubmitField("Predict")


# function to compute descriptors
def compute_morgan_fingerprint(smile):
    """Computes Morgan fingerprints for a given SMILES string."""
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None  # Return None for invalid SMILES

    mols = [mol]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols]
    fp_array = [np.array(fp) for fp in fps]
    column_names = [f"morgan_{i}" for i in range(2048)]
    return pd.DataFrame(fp_array, columns=column_names)


# --- Routes ---
@app.route("/")
def home():
    """Home page."""
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Prediction page with form."""
    form = PredictionForm()
    # Dynamically populate model choices
    models_path = os.path.join(os.path.dirname(__file__), "models")
    available_models = [f for f in os.listdir(models_path) if f.endswith(".pkl")]
    form.model.choices = [(model, model.split("_")[0]) for model in available_models]

    if form.validate_on_submit():
        smiles = form.smiles.data
        selected_model = form.model.data

        # Compute descriptors and handle invalid SMILES
        descriptors = compute_morgan_fingerprint(smiles)
        if descriptors is None:
            flash("Invalid SMILES string provided. Please try again.", "danger")
            return redirect(url_for("predict"))

        # Ensure descriptors is a 2D numpy array (n_samples, n_features)
        import numpy as np
        descriptors = np.asarray(descriptors)
        if descriptors.ndim == 1:
            descriptors = descriptors.reshape(1, -1)

        # Load the selected model
        model_path = os.path.join(os.path.dirname(__file__), "models", selected_model)
        print("Start to load model")
        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"Error loading model {e}")
            flash(f"Error loading model: {e}", "danger")
            return redirect(url_for("predict"))

        print(model)

        # Predict class (numeric) and map to label
        try:
            pred_arr = model.predict(descriptors)
            prediction_class = int(pred_arr[0])  # numeric 0/1
        except Exception as e:
            print(f"Error during prediction: {e}")
            flash(f"Error during prediction: {e}", "danger")
            return redirect(url_for("predict"))

        label_map = {0: "Inactive", 1: "Active"}
        prediction_label = label_map.get(prediction_class, str(prediction_class))

        # Get probability for the predicted class (best-effort)
        inhibition_prob = None
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(descriptors)[0]
                if 0 <= prediction_class < len(probs):
                    inhibition_prob = float(probs[prediction_class])
            elif hasattr(model, "decision_function"):
                score = model.decision_function(descriptors)[0]
                # binary -> score is a scalar; multiclass -> array
                if np.ndim(score) == 0:
                    # convert score -> probability via sigmoid
                    inhibition_prob = float(1.0 / (1.0 + np.exp(-score)))
                else:
                    # multiclass: softmax
                    exp = np.exp(score - np.max(score))
                    probs = exp / exp.sum()
                    if 0 <= prediction_class < len(probs):
                        inhibition_prob = float(probs[prediction_class])
        except Exception as e:
            # If probability computation fails, log it but continue returning the label
            print(f"Warning: could not compute probability: {e}")
            inhibition_prob = None

        # Format confidence for UI (keep as 0-1 rounded, or empty string if unavailable)
        confidence = round(inhibition_prob, 2) if inhibition_prob is not None else ""

        return redirect(
            url_for(
                "results",
                smiles=smiles,
                model=selected_model,
                prediction=prediction_label,  # "Active" or "Inactive"
                confidence=confidence,
            )
        )

    return render_template("predict.html", form=form)


@app.route("/results")
def results():
    """Display prediction results."""
    smiles = request.args.get("smiles")
    model = request.args.get("model")
    prediction = request.args.get("prediction")
    confidence = request.args.get("confidence")
    return render_template(
        "results.html",
        smiles=smiles,
        model=model,
        prediction=prediction,
        confidence=confidence,
    )


@app.errorhandler(404)
def page_not_found(e):
    """404 error handler."""
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_server_error(e):
    """500 error handler."""
    return render_template("500.html"), 500

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

