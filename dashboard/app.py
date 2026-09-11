from flask import Flask, jsonify, render_template

from dashboard.dashboard_data import (
    get_self_correction,
    get_system_summary,
    get_anomalies,
    get_drift,
    get_retraining,
    get_models,
    get_model_evaluation,
    get_dashboard_data
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder="templates"
)


# ============================================================
# DASHBOARD PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("dashboard.html")


# ============================================================
# API ROUTES
# ============================================================

@app.route("/api/system")
def system():
    return jsonify(
        get_system_summary()
    )


@app.route("/api/anomalies")
def anomalies():
    return jsonify(
        get_anomalies()
    )


@app.route("/api/drift")
def drift():
    return jsonify(
        get_drift()
    )


@app.route("/api/retraining")
def retraining():
    return jsonify(
        get_retraining()
    )


@app.route("/api/models")
def models():
    return jsonify(
        get_models()
    )


@app.route("/api/evaluation")
def evaluation():
    return jsonify(
        get_model_evaluation()
    )

@app.route("/api/self-correction")
def self_correction():

    return jsonify(
        get_self_correction()
    )

@app.route("/api/dashboard")
def dashboard():
    return jsonify(
        get_dashboard_data()
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print("PHASE 15 - NETWORK INTRUSION DETECTION DASHBOARD")
    print("=" * 80)

    print()
    print("Dashboard:")
    print("http://127.0.0.1:5000")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )