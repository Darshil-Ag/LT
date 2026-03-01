def calc_structural(allowed_stress, measured_stress, allowed_settlement, measured_settlement, allowed_deflection, actual_deflection):
    try:
        stress_ratio = measured_stress / allowed_stress if allowed_stress else 0
        settlement_ratio = measured_settlement / allowed_settlement if allowed_settlement else 0
        deflection_ratio = actual_deflection / allowed_deflection if allowed_deflection else 0
    except ZeroDivisionError:
        stress_ratio = settlement_ratio = deflection_ratio = 0

    risk_pct = ((stress_ratio + settlement_ratio + deflection_ratio) / 3) * 100
    risk_pct = round(risk_pct, 2)
    
    if risk_pct < 80:
        classification = "Safe"
    elif risk_pct <= 100:
        classification = "Warning"
    else:
        classification = "High Risk"

    mitigations = []
    if classification == "High Risk":
        mitigations = [
            "Increase beam depth",
            "Add reinforcement",
            "Add intermediate column",
            "Upgrade material grade"
        ]

    return {
        "stress_ratio": round(stress_ratio, 2),
        "settlement_ratio": round(settlement_ratio, 2),
        "deflection_ratio": round(deflection_ratio, 2),
        "risk_pct": risk_pct,
        "classification": classification,
        "mitigations": mitigations
    }

def calc_geotechnical(base_capacity, soil_factor, actual_load):
    updated_capacity = base_capacity * soil_factor
    
    if updated_capacity > 0 and actual_load > updated_capacity:
        risk_pct = (actual_load / updated_capacity) * 100
    else:
        risk_pct = 0
        
    risk_pct = round(risk_pct, 2)
    classification = "Safe"
    mitigations = []
    
    if risk_pct > 0:
        classification = "High Risk" if risk_pct > 100 else "Warning"
        mitigations = [
            "Increase pile depth",
            "Switch to raft foundation",
            "Soil stabilization",
            "Add drainage"
        ]

    return {
        "updated_capacity": round(updated_capacity, 2),
        "risk_pct": risk_pct,
        "classification": classification,
        "mitigations": mitigations
    }

def calc_schedule(planned_progress, actual_progress, elapsed_days):
    delay_pct = planned_progress - actual_progress
    delay_rate = delay_pct / elapsed_days if elapsed_days > 0 else 0
    risk_pct = delay_pct * 2
    
    delay_pct = round(delay_pct, 2)
    delay_rate = round(delay_rate, 2)
    risk_pct = round(max(0, risk_pct), 2)
    
    classification = "On Track"
    mitigations = []
    
    if actual_progress < planned_progress:
        classification = "Delayed"
        mitigations = [
            "Add extra shift",
            "Reallocate manpower",
            "Fast-track critical tasks",
            "Switch supplier"
        ]
        
    return {
        "delay_pct": delay_pct,
        "delay_rate": delay_rate,
        "risk_pct": risk_pct,
        "classification": classification,
        "mitigations": mitigations
    }

def calc_quality(required_strength, actual_strength, alignment_error, threshold):
    try:
        strength_ratio = actual_strength / required_strength if required_strength else 0
        alignment_ratio = alignment_error / threshold if threshold else 0
        inv_strength = (1 / strength_ratio) if strength_ratio > 0 else 1
    except ZeroDivisionError:
        strength_ratio = alignment_ratio = inv_strength = 0
        
    risk_pct = ((inv_strength + alignment_ratio) / 2) * 100
    risk_pct = round(risk_pct, 2)
    
    classification = "Good"
    mitigations = []
    
    if actual_strength < required_strength or alignment_error > threshold:
        classification = "Poor Quality"
        mitigations = [
            "Increase curing period",
            "Recalibrate structure",
            "Stop next-stage work",
            "Redesign load distribution"
        ]

    return {
        "strength_ratio": round(strength_ratio, 2),
        "alignment_ratio": round(alignment_ratio, 2),
        "risk_pct": risk_pct,
        "classification": classification,
        "mitigations": mitigations
    }

def calc_cost(extra_steel, steel_rate, extra_concrete, concrete_rate, delay_days, daily_cost):
    material_cost = (extra_steel * steel_rate) + (extra_concrete * concrete_rate)
    delay_cost = delay_days * daily_cost
    total_cost = material_cost + delay_cost
    
    return {
        "material_cost": round(material_cost, 2),
        "delay_cost": round(delay_cost, 2),
        "total_extra_cost": round(total_cost, 2)
    }

def calc_overall(structural_risk, geotech_risk, schedule_risk, quality_risk):
    total_risk = (0.4 * structural_risk) + (0.25 * geotech_risk) + (0.2 * schedule_risk) + (0.15 * quality_risk)
    return round(total_risk, 2)
