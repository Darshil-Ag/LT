# app.py
from flask import Flask, request, render_template, send_file, jsonify
import os
import json
import pickle
from werkzeug.utils import secure_filename
from trace_to_dxf import trace_image_to_dxf_with_text
from PIL import Image
import base64
from io import BytesIO

from construction_intelligence import (
    extract_layout_features,
    create_baseline_model,
    compute_live_features,
    encode_soil_type,
    mitigation_engine,
    generate_design_alternatives
)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
DATASET_FOLDER = 'dataset'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

import numpy as np

# Load the JSON data for plan similarity search
with open(os.path.join(DATASET_FOLDER, 'room_vectors_with_area.json'), 'r') as f:
    data = json.load(f)

# Track the most recently used plan metadata for downstream construction analytics.
CURRENT_PLAN_METADATA = {
    "total_area": 0.0,
    "room_counts": {},
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RISK_MODEL_PATH = os.path.join(BASE_DIR, "risk_model.pkl")

try:
    with open(RISK_MODEL_PATH, "rb") as f:
        risk_model = pickle.load(f)
except FileNotFoundError:
    risk_model = None
    app.logger.warning("Risk model not found at %s; /analyze_construction will be disabled.", RISK_MODEL_PATH)

def calculate_distance(input_counts, input_areas, entry_counts, entry_areas):
    # Calculate the difference in room counts
    counts_diff = 0
    all_rooms = set(input_counts.keys()).union(set(entry_counts.keys()))
    for room in all_rooms:
        counts_diff += (input_counts.get(room, 0) - entry_counts.get(room, 0)) ** 2
    
    # Calculate the difference in areas
    areas_diff = 0
    for room in all_rooms:
        areas_diff += (input_areas.get(room, 0) - entry_areas.get(room, 0)) ** 2

    # Return the combined distance (MSE for counts and areas)
    return counts_diff + areas_diff

def find_closest_match(user_input):
    input_counts = user_input.get("counts", {})
    input_areas = user_input.get("areas", {})

    closest_image = None
    closest_entry = None
    min_distance = float('inf')

    # Compare the user input with each entry in the data
    for entry in data:
        entry_counts = entry["input"].get("counts", {})
        entry_areas = entry["input"].get("areas", {})
        
        distance = calculate_distance(input_counts, input_areas, entry_counts, entry_areas)
        
        if distance < min_distance:
            min_distance = distance
            closest_image = entry["image"]
            closest_entry = entry

    return closest_image, closest_entry

def find_model_image(image_name):
    # Replace 'img.png' with 'model.png' to find the corresponding model image
    model_image_name = image_name.replace("img.png", "model.png")
    
    # Define the path to the rendered_images folder
    model_image_path = os.path.join(DATASET_FOLDER, "rendered_pngs", model_image_name)
    
    # Check if the model image exists
    if os.path.exists(model_image_path):
        return model_image_path
    else:
        return None  # Return None if the model image is not found

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/find_plan', methods=['POST'])
def find_plan():
    try:
        # Get JSON data from request
        user_input = request.json
        
        # Find the closest matching image and its metadata
        closest_image, closest_entry = find_closest_match(user_input)
        
        if not closest_image:
            return jsonify({"error": "No matching floor plan found"}), 404
        
        # Find the model image
        model_image_path = find_model_image(closest_image)
        
        if not model_image_path:
            return jsonify({"error": "Model image not found"}), 404

        # Persist plan metadata for construction intelligence layer
        entry_counts = closest_entry["input"].get("counts", {}) if closest_entry else {}
        entry_areas = closest_entry["input"].get("areas", {}) if closest_entry else {}
        total_area = float(sum(entry_areas.values())) if entry_areas else 0.0
        global CURRENT_PLAN_METADATA
        CURRENT_PLAN_METADATA = {
            "total_area": total_area,
            "room_counts": entry_counts,
        }
        
        # Load the image and convert to base64 for preview
        with open(model_image_path, "rb") as img_file:
            img_data = img_file.read()
        
        # Save the image to upload folder for processing
        input_path = os.path.join(UPLOAD_FOLDER, f"recommended_{os.path.basename(model_image_path)}")
        with open(input_path, "wb") as f:
            f.write(img_data)
        
        # Create a unique name for the output DXF file
        output_filename = f"plan_{os.path.basename(model_image_path).split('.')[0]}.dxf"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Process the image to DXF
        trace_image_to_dxf_with_text(input_path, output_path)
        
        # Return the image preview and DXF download path
        base64_img = base64.b64encode(img_data).decode('utf-8')
        
        return jsonify({
            "success": True,
            "image_preview": f"data:image/png;base64,{base64_img}",
            "dxf_path": f"/download/{output_filename}"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    image = request.files['image']
    
    if image.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if image:
        filename = secure_filename(image.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        output_filename = f"{filename.rsplit('.', 1)[0]}.dxf"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        image.save(input_path)
        
        try:
            trace_image_to_dxf_with_text(input_path, output_path)
            return jsonify({
                "success": True,
                "dxf_path": f"/download/{output_filename}"
            })
        except Exception as e:
            return jsonify({"error": f"Processing error: {str(e)}"}), 500
    
    return jsonify({"error": "Unknown error"}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=True)


@app.route('/analyze_construction', methods=['POST'])
def analyze_construction():
    if risk_model is None:
        return jsonify({"error": "Risk model is not available on the server."}), 503

    payload = request.get_json(force=True, silent=True) or {}

    soil_type = payload.get("soil_type", "")
    seismic_zone = payload.get("seismic_zone", "")
    floors = int(payload.get("floors", 1) or 1)
    cost_per_sqft = float(payload.get("cost_per_sqft", 0.0) or 0.0)
    live_data = payload.get("live_data") or {}

    # Derive layout features from the most recent plan metadata.
    layout_features = extract_layout_features(CURRENT_PLAN_METADATA)
    baseline_model = create_baseline_model(layout_features, soil_type, seismic_zone, floors)
    live_features = compute_live_features(baseline_model, live_data)

    # Assemble ML feature vector in the same order used during training.
    ml_features = [
        layout_features.get("total_area", 0.0),
        layout_features.get("max_span", 0.0),
        encode_soil_type(soil_type),
        float(floors),
        live_features.get("stress_ratio", 0.0),
        live_features.get("bearing_ratio", 1.0),
        live_features.get("rainfall", 0.0),
        live_features.get("progress_gap", 0.0),
    ]

    risk_score = float(risk_model.predict([ml_features])[0])
    risk_score = max(0.0, min(risk_score, 100.0))

    if risk_score < 35.0:
        risk_level = "Stable"
    elif risk_score < 70.0:
        risk_level = "At-Risk"
    else:
        risk_level = "Critical Intervention Required"

    mitigation_dicts = mitigation_engine(live_features, baseline_model)
    mitigations = [m["action"] for m in mitigation_dicts]

    # Compute cost and schedule from mitigations!
    cost_impact_pct = sum(m["cost_pct"] for m in mitigation_dicts)
    schedule_impact_days = sum(m["schedule_days"] for m in mitigation_dicts)
    cost_impact = f"+{cost_impact_pct:.1f}%"

    # Identify main driver of risk based on features
    risk_factors = {
        "stress_ratio": round(live_features.get("stress_ratio", 0.0), 2),
        "bearing_ratio": round(live_features.get("bearing_ratio", 1.0), 2),
        "progress_gap": round(live_features.get("progress_gap", 0.0), 2),
        "rainfall": round(live_features.get("rainfall", 0.0), 2)
    }
    
    main_driver = "None"
    max_severity = 0.0
    
    # Check severity
    if risk_factors["stress_ratio"] > 1.0:
        severity = risk_factors["stress_ratio"] - 1.0
        if severity > max_severity:
            max_severity = severity
            main_driver = "Beam stress exceeds allowed limits."
            
    if risk_factors["bearing_ratio"] < 0.85:
        severity = (0.85 - risk_factors["bearing_ratio"]) / 0.85
        if severity > max_severity:
            max_severity = severity
            main_driver = "Foundation bearing reduction."
            
    if risk_factors["progress_gap"] > 0.15:
        severity = (risk_factors["progress_gap"] - 0.15) / 0.15
        if severity > max_severity:
            max_severity = severity
            main_driver = "Significant schedule delay."
            
    if risk_factors["rainfall"] > 80.0:
        severity = (risk_factors["rainfall"] - 80.0) / 80.0
        if severity > max_severity:
            max_severity = severity
            main_driver = "Severe weather conditions."

    if max_severity == 0.0:
        main_driver = "Routine monitoring."

    # Generate dynamic risk explanation
    explanation = "All structural and geotechnical parameters remain within safe operating limits."
    if risk_factors["stress_ratio"] > 1.0:
        explanation = f"Measured beam stress is {risk_factors['stress_ratio']:.2f}x the allowable limit, indicating potential structural overstress under current loading conditions."
    elif risk_factors["bearing_ratio"] < 0.85:
        explanation = f"Bearing capacity has reduced to {risk_factors['bearing_ratio']:.2f}x expected value, increasing foundation instability risk."
    elif risk_factors["progress_gap"] > 0.15:
        progress_pct = risk_factors["progress_gap"] * 100
        explanation = f"Project progress is lagging by {progress_pct:.1f}% compared to plan."
    elif risk_factors["rainfall"] > 80.0:
        explanation = f"Severe weather recorded ({risk_factors['rainfall']:.1f} mm), pausing critical exterior pathways."

    # Phase 1: Risk Severity Index Refinement
    stress_severity = max(0.0, risk_factors["stress_ratio"] - 1.0)
    bearing_severity = max(0.0, 0.85 - risk_factors["bearing_ratio"])
    progress_severity = max(0.0, risk_factors["progress_gap"] - 0.1)
    rainfall_severity = risk_factors["rainfall"] / 200.0

    # Normalize roughly to 0-1 scale bounds
    risk_breakdown = {
        "structural": min(1.0, stress_severity / 0.5), # Caps if > 1.5x allowable
        "geotechnical": min(1.0, bearing_severity / 0.85),
        "schedule": min(1.0, progress_severity / 0.5),
        "environmental": min(1.0, rainfall_severity)
    }

    # Phase 2: Confidence Score for ML
    model_confidence = "Medium"
    if hasattr(risk_model, 'estimators_'):
        try:
            # Gather predictions from all individual trees in the forest
            tree_preds = [tree.predict([ml_features])[0] for tree in risk_model.estimators_]
            std_dev = np.std(tree_preds)
            if std_dev < 5.0:
                model_confidence = "High"
            elif std_dev > 15.0:
                model_confidence = "Low"
        except Exception:
            pass # fallback to Medium if tree traversal fails

    # Phase 4: Scenario Comparison Mode
    post_mitigation_risk_score = risk_score
    risk_reduction = 0.0
    if mitigation_dicts and len(mitigation_dicts) > 0:
        # Simulate top mitigation
        top_mitigation = mitigation_dicts[0]["action"]
        simulated_features = list(ml_features)
        if "beam" in top_mitigation.lower():
            # Assume 15% stress reduction
            simulated_features[4] = max(0.5, ml_features[4] * 0.85)
        elif "foundation" in top_mitigation.lower() or "soil" in top_mitigation.lower():
            # Assume 20% bearing improvement
            simulated_features[5] = min(1.5, ml_features[5] * 1.2)
        elif "schedule" in top_mitigation.lower() or "manpower" in top_mitigation.lower():
            # Assume progress gap halving
            simulated_features[7] = max(0.0, ml_features[7] * 0.5)

        post_score = float(risk_model.predict([simulated_features])[0])
        post_mitigation_risk_score = max(0.0, min(post_score, 100.0))
        risk_reduction = max(0.0, risk_score - post_mitigation_risk_score)

    # Phase 5: Risk Trend Simulation
    # Predict the score going forward if the current progress gap trend compounds
    risk_projection = [round(risk_score)]
    current_sim_features = list(ml_features)
    for _ in range(4):
        # Progress gap worsens by 2% each period if unmitigated
        current_sim_features[7] = min(0.5, current_sim_features[7] + 0.02)
        # Re-predict
        proj_score = float(risk_model.predict([current_sim_features])[0])
        proj_score = max(0.0, min(proj_score, 100.0))
        risk_projection.append(round(proj_score))

    # Phase 6: Executive Summary Generator
    if risk_score >= 70.0:
        summary_severity = "Critical Intervention Required"
    elif risk_score >= 35.0:
        summary_severity = "At-Risk"
    else:
        summary_severity = "Stable"

    primary_concerns = []
    if risk_factors["stress_ratio"] > 1.0:
        primary_concerns.append("structural overstress")
    if risk_factors["bearing_ratio"] < 0.85:
        primary_concerns.append("reduced bearing capacity")
    if risk_factors["progress_gap"] > 0.15:
        primary_concerns.append("moderate schedule lag")
    if risk_factors["rainfall"] > 80.0:
        primary_concerns.append("severe environmental delays")

    if not primary_concerns:
        concern_str = "no major deviations"
    elif len(primary_concerns) == 1:
        concern_str = primary_concerns[0]
    else:
        concern_str = f"{', '.join(primary_concerns[:-1])} and {primary_concerns[-1]}"

    if mitigation_dicts:
        top_rec = mitigation_dicts[0]["action"].lower().replace(".", "")
        recommendation = f"Immediate {top_rec} is recommended to prevent escalation."
    else:
        recommendation = "Continue routine monitoring as planned."

    executive_summary = f"Under current measurements, the project is classified as {summary_severity}. Primary concern arises from {concern_str}. {recommendation}"

    # Phase 7: Safety Margin Visualization
    # Margin = (allowed_stress - beam_stress) / allowed_stress
    # if stress_ratio is 0.8, margin is 20%
    allowed_stress = baseline_model.get("allowed_beam_stress", 1.0) or 1.0
    measured_stress = float(live_data.get("beam_stress", allowed_stress * risk_factors["stress_ratio"]))
    margin_ratio = max(0.0, 1.0 - risk_factors["stress_ratio"])
    safety_margin_pct = round(margin_ratio * 100, 1)

    # Phase 6: Generative Design Module hook
    generative_design = generate_design_alternatives(
        live_features=live_features,
        baseline_model=baseline_model,
        base_ml_features=ml_features,
        risk_model=risk_model,
        current_cost_pct=cost_impact_pct,
        current_schedule_days=schedule_impact_days
    )

    return jsonify(
        {
            "risk_score": round(risk_score, 1),
            "risk_level": risk_level,
            "mitigations": mitigations,
            "cost_impact": cost_impact,
            "schedule_impact_days": schedule_impact_days,
            "risk_factors": risk_factors,
            "main_driver": main_driver,
            "risk_explanation": explanation,
            "risk_breakdown": risk_breakdown,
            "model_confidence": model_confidence,
            "post_mitigation_risk_score": round(post_mitigation_risk_score, 1),
            "risk_reduction": round(risk_reduction, 1),
            "risk_projection": risk_projection,
            "executive_summary": executive_summary,
            "allowed_stress": round(allowed_stress, 1),
            "measured_stress": round(measured_stress, 1),
            "safety_margin_pct": safety_margin_pct,
            "generative_design": generative_design
        }
    )


@app.route('/run_pipeline', methods=['POST'])
def run_pipeline_endpoint():
    # Accept plan metadata and context in JSON body and run the orchestration.
    payload = request.get_json(force=True, silent=True) or {}
    plan_metadata = payload.get('plan_metadata') or CURRENT_PLAN_METADATA
    soil_type = payload.get('soil_type', 'sand')
    seismic_zone = payload.get('seismic_zone', 'III')
    floors = int(payload.get('floors', 1) or 1)
    cost_per_sqft = float(payload.get('cost_per_sqft', 0.0) or 0.0)
    steps = int(payload.get('steps', 6) or 6)

    report = run_pipeline(plan_metadata, soil_type=soil_type, seismic_zone=seismic_zone, floors=floors, cost_per_sqft=cost_per_sqft, steps=steps)
    return jsonify({"success": True, "report_summary": {"timeline_steps": len(report.get('timeline', []))}})


@app.route('/pipeline_report', methods=['GET'])
def pipeline_report():
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'pipeline_report.json')
    if not os.path.exists(report_path):
        return jsonify({"error": "Report not found"}), 404
    with open(report_path, 'r') as f:
        report = json.load(f)
    return jsonify(report)


@app.route('/dashboard', methods=['GET'])
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)
