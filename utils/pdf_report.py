from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph

def generate_pdf(filename, disease, confidence, description, treatment, prevention):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("<b>Mango Disease Prediction Report</b>", styles["Title"]))
    elements.append(Paragraph(f"<b>Disease:</b> {disease}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles["Normal"]))
    elements.append(Paragraph(f"<b>Description:</b> {description}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Treatment:</b> {treatment}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Prevention:</b> {prevention}", styles["Normal"]))

    doc.build(elements)