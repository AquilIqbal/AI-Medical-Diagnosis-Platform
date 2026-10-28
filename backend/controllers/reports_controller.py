"""Template-rendering controllers for the Reports module."""

from flask import flash, redirect, render_template, request, url_for

from backend.services.diagnosis_service import DiagnosisService


reports_service = DiagnosisService()


def get_confidence_level(confidence_score: float | None) -> str:
    """Return the confidence interpretation label for a model score.
    
    Args:
        confidence_score: Model confidence as a decimal (0.0 to 1.0)
    
    Returns:
        Confidence level label: "Very High Confidence", "High Confidence", etc.
    """
    if confidence_score is None:
        return "Not available"
    
    percentage = confidence_score * 100
    if percentage >= 90:
        return "Very High Confidence"
    elif percentage >= 75:
        return "High Confidence"
    elif percentage >= 60:
        return "Moderate Confidence"
    else:
        return "Low Confidence"


def build_overall_ai_assessment(diagnosis) -> dict:
    """Build overall AI assessment data from available predictions.
    
    Returns a dictionary with:
    - has_predictions: whether any predictions exist
    - predictions: list of {type, label, confidence_percent} for each modality
    - summary_text: description of available results
    """
    nlp_pred = None
    clinical_pred = None
    cnn_pred = None
    
    for pred in diagnosis.ai_predictions:
        if pred.prediction_type == "nlp":
            nlp_pred = pred
        elif pred.prediction_type == "clinical_ml":
            clinical_pred = pred
        elif pred.prediction_type == "cnn":
            cnn_pred = pred
    
    predictions = []
    
    if diagnosis.predicted_disease:
        predictions.append({
            "type": "NLP",
            "label": diagnosis.predicted_disease,
            "confidence_percent": f"{diagnosis.confidence_score * 100:.1f}%" if diagnosis.confidence_score is not None else "N/A"
        })
    
    if clinical_pred:
        predictions.append({
            "type": "Clinical ML",
            "label": clinical_pred.predicted_label,
            "confidence_percent": f"{clinical_pred.confidence_score * 100:.1f}%" if clinical_pred.confidence_score is not None else "N/A"
        })
    
    if cnn_pred:
        predictions.append({
            "type": "CNN",
            "label": cnn_pred.predicted_label,
            "confidence_percent": f"{cnn_pred.confidence_score * 100:.1f}%" if cnn_pred.confidence_score is not None else "N/A"
        })
    
    has_predictions = len(predictions) > 0
    
    if has_predictions:
        summary_text = "Available AI models produced the following results:"
    else:
        summary_text = "No AI prediction results are currently available for this diagnosis."
    
    return {
        "has_predictions": has_predictions,
        "predictions": predictions,
        "summary_text": summary_text
    }


def generate_clinical_recommendations(diagnosis) -> dict:
    """Generate dynamic clinical recommendations based on actual predictions.
    
    Returns a dictionary with:
    - key_findings: list of findings from each modality
    - next_steps: list of recommended next steps
    """
    key_findings = []
    next_steps = []
    
    nlp_pred = None
    clinical_pred = None
    cnn_pred = None
    
    for pred in diagnosis.ai_predictions:
        if pred.prediction_type == "nlp":
            nlp_pred = pred
        elif pred.prediction_type == "clinical_ml":
            clinical_pred = pred
        elif pred.prediction_type == "cnn":
            cnn_pred = pred
    
    # Collect findings from each modality
    if diagnosis.predicted_disease:
        key_findings.append(f"NLP symptom analysis: {diagnosis.predicted_disease}")
    
    if clinical_pred:
        key_findings.append(f"Clinical ML assessment: {clinical_pred.predicted_label}")
    
    if cnn_pred:
        key_findings.append(f"CNN image analysis: {cnn_pred.predicted_label}")
    
    # Generate contextual recommendations based on what's available
    has_nlp = diagnosis.predicted_disease is not None
    has_clinical_ml = clinical_pred is not None
    has_cnn = cnn_pred is not None
    
    if has_nlp:
        next_steps.append("Review the NLP symptom classification alongside the patient's reported symptoms and clinical examination.")
    
    if has_cnn:
        next_steps.append("Review the CNN image classification and Grad-CAM visualization with qualified clinical/radiology personnel.")
    
    if has_clinical_ml:
        if clinical_pred.predicted_label == "Abnormal":
            next_steps.append("Review the clinical risk assessment identifying abnormal findings with a qualified healthcare professional.")
        elif clinical_pred.predicted_label == "Normal":
            next_steps.append("Clinical ML assessment was classified as Normal; continue interpretation in the context of the complete clinical presentation.")
    
    # Multi-modality specific recommendations
    modality_count = sum([has_nlp, has_clinical_ml, has_cnn])
    if modality_count > 1:
        # Check for disagreement between modalities
        labels = set()
        if has_nlp:
            labels.add(diagnosis.predicted_disease)
        if has_cnn:
            labels.add(cnn_pred.predicted_label)
        
        if len(labels) > 1 and has_nlp and has_cnn:
            next_steps.append("AI model outputs are not fully concordant and require clinical review and correlation with clinical findings.")
        elif modality_count >= 2:
            next_steps.append("Compare the outputs across available modalities and assess whether they are concordant or discordant.")
            next_steps.append("Correlate AI outputs with clinical findings and established diagnostic procedures.")
    
    # Ensure we always have at least one general recommendation
    if not next_steps:
        next_steps.append("Review all available clinical information with qualified healthcare professionals.")
    
    next_steps.append("Document clinical decisions and follow-up procedures in the patient record.")
    
    return {
        "key_findings": key_findings,
        "next_steps": next_steps
    }


def list_reports() -> str:
    """Render reports, optionally filtered by a search query."""
    search_query = request.args.get("search", "").strip()
    diagnoses = reports_service.get_all_diagnoses(search_query)
    return render_template(
        "reports/list.html",
        diagnoses=diagnoses,
        search_query=search_query,
    )


def view_report(diagnosis_id: int) -> str | object:
    """Render the full report for a specific diagnosis with all AI analysis."""
    diagnosis = reports_service.get_diagnosis_by_id(diagnosis_id)
    if diagnosis is None:
        flash("Report for the specified diagnosis was not found.", "error")
        return redirect(url_for("reports.list_reports"))

    # Prepare data for template
    overall_assessment = build_overall_ai_assessment(diagnosis)
    recommendations = generate_clinical_recommendations(diagnosis)
    
    return render_template(
        "reports/detail.html",
        diagnosis=diagnosis,
        get_confidence_level=get_confidence_level,
        overall_assessment=overall_assessment,
        recommendations=recommendations
    )
