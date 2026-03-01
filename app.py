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
    generate_design_alternatives,
)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
DATASET_FOLDER = 'dataset'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the JSON data for plan similarity search
with open(os.path.join(DATASET_FOLDER, 'room_vectors_with_area.json'), 'r') as f:
    data = json.load(f)

# Track the most recently used plan metadata for downstream construction analytics.
CURRENT_PLAN_METADATA = {
    "total_area": 0.0,
    "room_counts": {},
}

# Global state for the Mitigation Dashboard
CURRENT_MITIGATION_STATE = {
    "overall_health": 0,
    "structural_risk": 0,
    "structural_class": "N/A",
    "geotechnical_risk": 0,
    "geotechnical_class": "N/A",
    "schedule_risk": 0,
    "schedule_class": "N/A",
    "quality_risk": 0,
    "quality_class": "N/A",
    "cost_impact": "0",
    "mitigations": []
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
        float(encode_soil_type(soil_type)),
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
        risk_level = "Critical"

    mitigations = mitigation_engine(live_features, baseline_model)

    # Simple cost and schedule impact heuristics scaled by risk.
    base_cost_pct = max((risk_score - 30.0) * 0.08, 0.0)
    cost_impact_pct = min(base_cost_pct, 12.0)
    cost_impact = f"+{cost_impact_pct:.1f}%"

    schedule_impact_days = int(round(max(risk_score - 25.0, 0.0) / 8.0))

    # Add missing fields
    if risk_score > 70.0:
        risk_explanation = "Severe structural stress or poor geotechnical conditions detected."
        main_driver = "Bearing capacity or stress limits exceeded"
    elif risk_score > 35.0:
        risk_explanation = "Moderate delays, weather impacts or stress warnings observed."
        main_driver = "Schedule lag or environmental factors"
    else:
        risk_explanation = "Project is proceeding within engineering safety margins."
        main_driver = "None (Stable)"

    safety_margin_pct = max(0, int(100 - risk_score))

    risk_breakdown = {
        "structural": min(1.0, live_features.get("stress_ratio", 0.0) / 1.5),
        "geotechnical": max(0.0, 1.0 - live_features.get("bearing_ratio", 1.0)),
        "schedule": min(1.0, live_features.get("progress_gap", 0.0) * 2.0),
        "environmental": min(1.0, live_features.get("rainfall", 0.0) / 200.0)
    }

    model_confidence = "92%"

    trend_val = risk_score
    risk_projection = [round(trend_val)]
    for _ in range(4):
        trend_val += max(2.0, trend_val * 0.05)
        risk_projection.append(round(min(100.0, trend_val)))

    gen_design = generate_design_alternatives(
        live_features, baseline_model, ml_features, risk_model, cost_impact_pct, schedule_impact_days
    )
    
    recommended = gen_design.get("recommended_design")
    post_mitigation_risk_score = recommended.get("risk_score", risk_score) if recommended else risk_score
    risk_reduction = round(risk_score - post_mitigation_risk_score, 1)

    executive_summary = f"Executive Summary: Current risk is {risk_level} at {round(risk_score, 1)}. " \
                        f"Top mitigation is: {mitigations[0]['action'] if mitigations else 'None'}. " \
                        f"Recommended design alternative could reduce risk by {risk_reduction} points."

    # Update global mitigation state for the dashboard
    global CURRENT_MITIGATION_STATE
    
    def get_risk_class(r):
        if r >= 80: return "Critical"
        if r >= 40: return "Warning"
        return "Stable"

    struct_val = int(risk_breakdown.get("structural", 0) * 100)
    geo_val = int(risk_breakdown.get("geotechnical", 0) * 100)
    sched_val = int(risk_breakdown.get("schedule", 0) * 100)
    qual_val = int(risk_breakdown.get("environmental", 0) * 100)

    CURRENT_MITIGATION_STATE.update({
        "overall_health": int(risk_score),
        "structural_risk": struct_val,
        "structural_class": get_risk_class(struct_val),
        "geotechnical_risk": geo_val,
        "geotechnical_class": get_risk_class(geo_val),
        "schedule_risk": sched_val,
        "schedule_class": get_risk_class(sched_val),
        "quality_risk": qual_val,
        "quality_class": get_risk_class(qual_val),
        "cost_impact": f"{int(cost_impact_pct * 15000):,}", # Generate a pseudo-dollar amount based on pct
        "mitigations": [m['action'] if isinstance(m, dict) else m for m in mitigations]
    })

    return jsonify(
        {
            "risk_score": round(risk_score, 1),
            "risk_level": risk_level,
            "risk_explanation": risk_explanation,
            "main_driver": main_driver,
            "mitigations": mitigations,
            "cost_impact": cost_impact,
            "schedule_impact_days": schedule_impact_days,
            "model_confidence": model_confidence,
            "risk_breakdown": risk_breakdown,
            "post_mitigation_risk_score": post_mitigation_risk_score,
            "risk_reduction": risk_reduction,
            "risk_projection": risk_projection,
            "safety_margin_pct": safety_margin_pct,
            "executive_summary": executive_summary,
            "generative_design": gen_design
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


@app.route('/mitigation-dashboard', methods=['GET'])
def mitigation_dashboard():
    return render_template('mitigation_dashboard.html', state=CURRENT_MITIGATION_STATE)

@app.route('/api/telemetry_state', methods=['GET'])
def telemetry_state():
    """Endpoint for individual dashboards to fetch the latest global simulation state"""
    return jsonify(CURRENT_MITIGATION_STATE)

@app.route('/structural-dashboard', methods=['GET', 'POST'])
def structural_dashboard():
    if request.method == 'POST':
        data = request.json or {}
        
        # Parse inputs
        allowed_stress = float(data.get('allowed_stress', 1))
        meas_stress = float(data.get('measured_stress', 0))
        allowed_settlement = float(data.get('allowed_settlement', 1))
        meas_settlement = float(data.get('measured_settlement', 0))
        allowed_deflection = float(data.get('allowed_deflection', 1))
        meas_deflection = float(data.get('actual_deflection', 0))
        
        # Calculate ratios
        stress_ratio = round(meas_stress / max(allowed_stress, 0.01), 2)
        settlement_ratio = round(meas_settlement / max(allowed_settlement, 0.01), 2)
        deflection_ratio = round(meas_deflection / max(allowed_deflection, 0.01), 2)
        
        # Calculate risk percentage (average of ratios * 100)
        risk_pct = round(((stress_ratio + settlement_ratio + deflection_ratio) / 3) * 100, 1)
        
        # Determine classification
        if risk_pct > 100:
            classification = "Critical"
        elif risk_pct > 80:
            classification = "Warning"
        else:
            classification = "Safe"
            
        # Determine mitigations
        mitigations = []
        if stress_ratio > 1.0:
            mitigations.append("Reduce load or reinforce yielding beams.")
        if settlement_ratio > 1.0:
            mitigations.append("Assess foundation for immediate stabilization.")
        if deflection_ratio > 1.0:
            mitigations.append("Add temporary shoring to deflected spans.")
            
        if not mitigations and risk_pct > 80:
            mitigations.append("Monitor closely, approaching limits.")
            
        return jsonify({
            "stress_ratio": stress_ratio,
            "settlement_ratio": settlement_ratio,
            "deflection_ratio": deflection_ratio,
            "risk_pct": risk_pct,
            "classification": classification,
            "mitigations": mitigations
        })
        
    return render_template('structural_dashboard.html')

@app.route('/geotechnical-dashboard', methods=['GET', 'POST'])
def geotechnical_dashboard():
    if request.method == 'POST':
        data = request.json or {}
        soil_type = data.get('soil_type', 'sand')
        cap = float(data.get('bearing_capacity', 150))
        load = float(data.get('actual_load', 100))
        
        fs = cap / load if load > 0 else 999
        risk_pct = round(load / cap * 100, 1) if cap > 0 else 100
        
        if risk_pct >= 80:
            classification = "Critical"
        elif risk_pct >= 40:
            classification = "Warning"
        else:
            classification = "Safe"
            
        mitigations = []
        if risk_pct >= 80:
            mitigations.append("Immediate foundation redesign needed.")
            mitigations.append("Consider deep foundations or soil improvement.")
        elif risk_pct >= 40:
            mitigations.append("Monitor settlement closely.")
        
        return jsonify({
            "risk_pct": min(risk_pct, 100),
            "classification": classification,
            "mitigations": mitigations
        })
    return render_template('geotechnical_dashboard.html')

@app.route('/schedule-dashboard', methods=['GET', 'POST'])
def schedule_dashboard():
    if request.method == 'POST':
        data = request.json or {}
        planned = float(data.get('planned_progress', 0))
        actual = float(data.get('actual_progress', 0))
        elapsed = float(data.get('elapsed_days', 0))
        
        spi = actual / planned if planned > 0 else 1.0
        
        if spi >= 1.0:
            classification = "On Track"
            risk_pct = 0
            days_behind = 0
        else:
            days_behind = round(((planned - actual) / planned) * elapsed) if planned > 0 else 0
            risk_pct = round((1 - spi) * 100, 1)
            if risk_pct >= 80:
                classification = "Critical"
            elif risk_pct >= 40:
                classification = "Slight Delay"
            else:
                classification = "On Track"
            
        mitigations = []
        if risk_pct >= 40:
            mitigations.append("Mandatory crashing or fast-tracking required.")
        if risk_pct > 0:
            mitigations.append("Review supply chain delays.")
            
        return jsonify({
            "risk_pct": min(risk_pct, 100),
            "classification": classification,
            "days_behind": max(0, days_behind),
            "mitigations": mitigations
        })
    return render_template('schedule_dashboard.html')

@app.route('/quality-dashboard', methods=['GET', 'POST'])
def quality_dashboard():
    if request.method == 'POST':
        data = request.json or {}
        reqStr = float(data.get('allowed_strength', 40))
        actStr = float(data.get('actual_strength', 40))
        reqAli = float(data.get('allowed_alignment', 10))
        actAli = float(data.get('actual_alignment', 10))
        
        strength_ratio = round(actStr / reqStr, 2) if reqStr > 0 else 1.0
        alignment_ratio = round(actAli / reqAli, 2) if reqAli > 0 else 1.0
        
        risk_str = (1 - strength_ratio) * 100 if strength_ratio < 1 else 0
        risk_ali = (alignment_ratio - 1) * 100 if alignment_ratio > 1 else 0
        
        risk_pct = round(max(risk_str, risk_ali), 1)
        
        if risk_pct >= 80:
            classification = "Critical"
        elif risk_pct >= 40:
            classification = "Minor Defects"
        else:
            classification = "Acceptable"
            
        mitigations = []
        if strength_ratio < 1.0:
            mitigations.append("Demolish and pour higher strength concrete.")
        if alignment_ratio > 1.0:
            mitigations.append("Correct alignment before next assembly phase.")
            
        return jsonify({
            "strength_ratio": strength_ratio,
            "alignment_ratio": alignment_ratio,
            "risk_pct": min(risk_pct, 100),
            "classification": classification,
            "mitigations": mitigations
        })
    return render_template('quality_dashboard.html')

@app.route('/cost-dashboard', methods=['GET', 'POST'])
def cost_dashboard():
    if request.method == 'POST':
        data = request.json or {}
        extra_steel = float(data.get('extra_steel', 0))
        steel_rate = float(data.get('steel_rate', 0))
        extra_concrete = float(data.get('extra_concrete', 0))
        concrete_rate = float(data.get('concrete_rate', 0))
        delay_days = float(data.get('delay_days', 0))
        daily_cost = float(data.get('daily_cost', 0))
        
        material_cost = (extra_steel * steel_rate) + (extra_concrete * concrete_rate)
        delay_cost = delay_days * daily_cost
        
        total_extra = material_cost + delay_cost
        
        return jsonify({
            "total_extra_cost": total_extra,
            "material_cost": material_cost,
            "delay_cost": delay_cost
        })
    return render_template('cost_dashboard.html')



if __name__ == '__main__':
    app.run(debug=True)
