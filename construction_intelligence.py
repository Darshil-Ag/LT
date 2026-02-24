import math
from typing import Dict, Any, List


def extract_layout_features(plan_metadata: Dict[str, Any]) -> Dict[str, float]:
    """Derive normalized layout features from basic plan metadata.

    plan_metadata is expected to contain:
      - total_area: total built-up area (sq ft or consistent unit)
      - room_counts: mapping of room_type -> count
      - optional overrides for span_length and wall_density
    """
    total_area = float(plan_metadata.get("total_area", 0.0)) or 0.0
    room_counts = plan_metadata.get("room_counts") or {}
    total_rooms = max(sum(room_counts.values()), 1)

    # Estimate a representative span length from area and room count.
    # Larger, more open layouts yield longer spans.
    # This is a coarse heuristic, not a code check.
    base_span = math.sqrt(max(total_area, 1.0))
    openness_factor = max(1.0 - (total_rooms / 30.0), 0.4)
    span_length = plan_metadata.get("span_length") or base_span * openness_factor

    # Approximate wall density: more rooms generally imply more internal walls.
    # Normalize into a 0.1 - 0.7 band for later scaling.
    raw_density = 0.15 + 0.03 * total_rooms
    wall_density = plan_metadata.get("wall_density") or max(0.1, min(raw_density, 0.7))

    # Max span is slightly higher than representative span to account for primary frames.
    max_span = span_length * 1.15

    return {
        "total_area": total_area,
        "total_rooms": float(total_rooms),
        "span_length": float(span_length),
        "wall_density": float(wall_density),
        "max_span": float(max_span),
    }


def _soil_base_capacity(soil_type: str) -> float:
    """Very coarse allowable bearing capacities (kN/m² or consistent relative unit)."""
    soil_type = (soil_type or "").strip().lower()
    if "rock" in soil_type:
        return 450.0
    if "gravel" in soil_type:
        return 350.0
    if "sand" in soil_type:
        return 250.0
    if "clay" in soil_type:
        return 180.0
    if "fill" in soil_type or "loose" in soil_type:
        return 120.0
    return 200.0


def _seismic_importance_factor(seismic_zone: str) -> float:
    """Return an importance factor based on seismic hazard."""
    z = (seismic_zone or "").strip().upper()
    if any(token in z for token in ("IV", "5", "HIGH")):
        return 1.3
    if any(token in z for token in ("III", "4", "MODERATE")):
        return 1.15
    return 1.0


def create_baseline_model(
    layout_features: Dict[str, float],
    soil_type: str,
    seismic_zone: str,
    floors: int,
) -> Dict[str, float]:
    """Create a rule-based baseline structural model from layout and context.

    The goal is to produce stable engineering-style targets, not code-precise values.
    """
    floors = max(int(floors), 1)
    total_area = layout_features.get("total_area", 0.0)
    span_length = layout_features.get("span_length", 0.0)

    # Allowable beam stress (e.g. MPa) – reduced with taller buildings and longer spans.
    base_beam_stress = 250.0
    height_factor = 1.0 - min((floors - 1) * 0.02, 0.3)
    span_factor = 1.0 - min((span_length - 5.0) * 0.01, 0.25) if span_length > 5 else 1.0
    allowed_beam_stress = base_beam_stress * height_factor * span_factor

    # Soil capacity adjusted by seismic importance.
    base_capacity = _soil_base_capacity(soil_type)
    importance_factor = _seismic_importance_factor(seismic_zone)
    expected_bearing_capacity = base_capacity / importance_factor

    # Settlement limit (mm) – tighter limits for higher seismicity and taller buildings.
    base_settlement_limit = 25.0
    settlement_reduce = (floors - 1) * 0.5 + (importance_factor - 1.0) * 10.0
    settlement_limit = max(15.0, base_settlement_limit - settlement_reduce)

    # Global safety factor – baseline adjusted by soil quality and height.
    safety_factor = 1.5
    if base_capacity < 180:
        safety_factor += 0.2
    if floors > 5:
        safety_factor += 0.1

    return {
        "allowed_beam_stress": float(allowed_beam_stress),
        "expected_bearing_capacity": float(expected_bearing_capacity),
        "settlement_limit": float(settlement_limit),
        "safety_factor": float(safety_factor),
        "floors": float(floors),
        "total_area": float(total_area),
    }


def compute_live_features(
    baseline_model: Dict[str, float],
    live_data: Dict[str, float],
) -> Dict[str, float]:
    """Derive live monitoring features relative to the baseline model."""
    allowed_beam_stress = baseline_model.get("allowed_beam_stress", 1.0) or 1.0
    expected_bearing_capacity = baseline_model.get("expected_bearing_capacity", 1.0) or 1.0
    settlement_limit = baseline_model.get("settlement_limit", 1.0) or 1.0

    beam_stress = float(live_data.get("beam_stress", 0.0))
    bearing_capacity = float(live_data.get("bearing_capacity", 0.0))
    settlement = float(live_data.get("settlement", 0.0))
    rainfall = float(live_data.get("rainfall", 0.0))
    planned_progress = max(float(live_data.get("planned_progress", 0.0)), 1.0)
    actual_progress = float(live_data.get("actual_progress", 0.0))
    concrete_strength = float(live_data.get("concrete_strength", 0.0))

    stress_ratio = beam_stress / allowed_beam_stress
    bearing_ratio = bearing_capacity / expected_bearing_capacity
    settlement_ratio = settlement / settlement_limit
    progress_gap = (planned_progress - actual_progress) / planned_progress
    progress_gap = max(progress_gap, 0.0)

    # Quality index combines material performance and deformation.
    strength_factor = min(concrete_strength / 25.0, 1.2)
    deformation_penalty = min(max(settlement_ratio - 1.0, 0.0) * 0.5, 0.6)
    quality_index = max(min(strength_factor - deformation_penalty, 1.2), 0.0)

    # Delay index grows with progress lag and adverse weather.
    delay_index = min(progress_gap * 1.5 + (rainfall / 300.0), 2.0)

    return {
        "stress_ratio": float(stress_ratio),
        "bearing_ratio": float(bearing_ratio),
        "settlement_ratio": float(settlement_ratio),
        "progress_gap": float(progress_gap),
        "quality_index": float(quality_index),
        "delay_index": float(delay_index),
        "rainfall": float(rainfall),
    }


def encode_soil_type(soil_type: str) -> int:
    """Stable integer encoding for soil types used in ML features."""
    soil_type = (soil_type or "").strip().lower()
    mapping = {
        "rock": 0,
        "gravel": 1,
        "sand": 2,
        "clay": 3,
        "fill": 4,
    }
    for key, code in mapping.items():
        if key in soil_type:
            return code
    return 5  # other / unknown


def mitigation_engine(
    live_features: Dict[str, float],
    baseline_model: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Generate human-readable mitigations with associated cost and schedule impacts."""
    mitigations: List[Dict[str, Any]] = []

    stress_ratio = live_features.get("stress_ratio", 0.0)
    bearing_ratio = live_features.get("bearing_ratio", 1.0)
    progress_gap = live_features.get("progress_gap", 0.0)
    rainfall = live_features.get("rainfall", 0.0)

    if stress_ratio > 1.0:
        mitigations.append({"action": "Increase beam depth or section size.", "cost_pct": 3.0, "schedule_days": 2})
        mitigations.append({"action": "Add longitudinal and shear reinforcement in critical beams.", "cost_pct": 2.5, "schedule_days": 1})

    if bearing_ratio < 0.85:
        mitigations.append({"action": "Increase foundation depth or width to reduce contact pressure.", "cost_pct": 4.0, "schedule_days": 5})
        mitigations.append({"action": "Introduce soil stabilization (e.g., compaction, grouting, or replacement).", "cost_pct": 3.5, "schedule_days": 4})

    if progress_gap > 0.15:
        mitigations.append({"action": "Add additional manpower or shifts on critical path activities.", "cost_pct": 2.0, "schedule_days": 0})
        mitigations.append({"action": "Fast-track or resequence critical path tasks to recover schedule.", "cost_pct": 1.0, "schedule_days": 0})

    if rainfall > 80.0:
        mitigations.append({"action": "Delay slab casting and exterior works until rainfall reduces.", "cost_pct": 0.5, "schedule_days": 3})
        mitigations.append({"action": "Provide temporary weather protection and drainage at site.", "cost_pct": 1.5, "schedule_days": 1})

    if not mitigations:
        mitigations.append({"action": "Continue as planned with routine monitoring; no immediate mitigation required.", "cost_pct": 0.0, "schedule_days": 0})

    # Phase 3: Mitigation Prioritization
    for m in mitigations:
        m["impact_score"] = m["cost_pct"] * 0.5 + m["schedule_days"] * 0.5

    mitigations.sort(key=lambda x: x["impact_score"], reverse=True)

    for i, m in enumerate(mitigations):
        m["priority_rank"] = i + 1

    return mitigations


def generate_design_alternatives(
    live_features: Dict[str, float],
    baseline_model: Dict[str, float],
    base_ml_features: List[float],
    risk_model: Any,
    current_cost_pct: float,
    current_schedule_days: int
) -> Dict[str, Any]:
    """
    Generates structural design alternatives by exploring parametric grid spaces.
    Evaluates risk and cost tradeoffs per alternative using ML simulation.
    """
    
    # Grid Search parameter sets
    beam_factors = [1.0, 1.1, 1.2]
    column_factors = [1.0, 1.15]
    foundation_factors = [1.0, 1.2]
    
    alternatives = []
    design_id = 1
    
    base_stress_ratio = live_features.get("stress_ratio", 0.0)
    base_bearing_ratio = live_features.get("bearing_ratio", 1.0)
    
    for bf in beam_factors:
        for cf in column_factors:
            for ff in foundation_factors:
                
                # Skip baseline copy
                if bf == 1.0 and cf == 1.0 and ff == 1.0:
                    continue
                    
                # Cap the combinatorial explosion
                if design_id > 10:
                    break
                    
                # 1. Compute physical structural adjustment
                # Beam depth expands allowable stress -> stress ratio drops
                adj_stress_ratio = base_stress_ratio / bf
                # Foundation expansion -> bearing ratio improves
                adj_bearing_ratio = base_bearing_ratio * ff
                
                # 2. Adjust ML specific feature array matching [area, span, soil, floors, stress, bearing, rain, gap]
                simulated_features = list(base_ml_features)
                simulated_features[4] = adj_stress_ratio
                simulated_features[5] = adj_bearing_ratio
                
                # 3. Simulate new ML predicted risk
                try:
                    alt_risk = float(risk_model.predict([simulated_features])[0])
                except Exception:
                    alt_risk = float(base_ml_features[4] * 100) # Fallback if model missing
                alt_risk = max(0.0, min(alt_risk, 100.0))
                
                # 4. Model cost and schedule penalties
                # Beam +2% per 0.1, Col +1.5% per 0.1, Foundation +3% per 0.1
                cost_impact = current_cost_pct
                cost_impact += ((bf - 1.0) * 10) * 2.0
                cost_impact += ((cf - 1.0) * 10) * 1.5
                cost_impact += ((ff - 1.0) * 10) * 3.0
                
                schedule_impact = current_schedule_days
                if bf > 1.0: schedule_impact += 1
                if cf > 1.0: schedule_impact += 1
                if ff > 1.0: schedule_impact += 2
                
                # 5. Compute optimization threshold (0.6 Risk + 0.4 Cost)
                # Lower score is heavily preferable
                opt_score = (alt_risk * 0.6) + (cost_impact * 5.0 * 0.4)
                
                alternatives.append({
                    "design_id": design_id,
                    "parameters": {
                        "beam_depth_factor": round(bf, 2),
                        "column_density_factor": round(cf, 2),
                        "foundation_depth_factor": round(ff, 2)
                    },
                    "risk_score": round(alt_risk, 1),
                    "cost_impact_pct": round(cost_impact, 1),
                    "schedule_days": schedule_impact,
                    "optimization_score": round(opt_score, 2),
                    "status": "Explored"
                })
                
                design_id += 1
                
    # Rank alternatives by optimization threshold (ascending)
    alternatives.sort(key=lambda x: x["optimization_score"])
    
    if alternatives:
        alternatives[0]["status"] = "Recommended"
        recommended_design = alternatives[0]
    else:
        recommended_design = None
        
    return {
        "recommended_design": recommended_design,
        "alternatives": alternatives
    }
