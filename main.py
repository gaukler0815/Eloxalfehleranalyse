
import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import base64
from fpdf import FPDF

# Fehlerklassen
CLASSES = {
    0: {
        "name": "Flecken / Ausgasung",
        "ursache": "Lufteinschlüsse oder Rückstände in der Vorbehandlung",
        "behebung": "Optimierung der Entfettung und Spülprozesse, Badparameter prüfen"
    },
    1: {
        "name": "Kontaktabdrücke",
        "ursache": "Ungünstige Kontaktierung oder beschädigte Gestelle",
        "behebung": "Verwendung von kontaktierten Stellen außerhalb der Sichtfläche, Gestellwartung"
    },
    2: {
        "name": "Verfärbung durch Legierung",
        "ursache": "Ungeeignete Legierung mit hohem Kupfer-/Zinkanteil",
        "behebung": "Bevorzugung von 6060 oder 6082, Beizzeiten anpassen"
    },
    3: {
        "name": "Grauschleier (7075)",
        "ursache": "Intermetallische Einschlüsse streuen Licht diffus",
        "behebung": "7075 vermeiden oder Aktivierung vor Anodisation, z. B. mit Salpetersäure"
    },
    4: {
        "name": "Vorbehandlungsfehler",
        "ursache": "Unvollständige Entfettung oder Beizfehler",
        "behebung": "Beize überprüfen, Reinigungszyklen optimieren"
    },
}

# Bildvorverarbeitung
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Modell laden
def load_model():
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

def classify_image(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        class_idx = torch.argmax(probabilities).item()
        confidence = probabilities[class_idx].item()
    return class_idx, confidence

def generate_pdf(reports):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Technischer Analysebericht - Eloxaloberfläche", ln=True, align='C')
    pdf.ln(10)
    for r in reports:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"Fehler: {r['fehler']}", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, f"Ursache: {r['ursache']}")
        pdf.multi_cell(0, 10, f"Empfehlung: {r['behebung']}")
        pdf.ln(5)
    return pdf.output(dest='S').encode('latin1')

st.title("KI-Fehleranalyse eloxierter Bauteile")
st.markdown("Laden Sie bis zu 5 Bilder hoch. Die Analyse erfolgt automatisch, ohne Datenspeicherung.")

view = st.radio("Darstellungsmodus wählen:", ["Kundenerklärung", "Detailbericht für QS"])
uploaded_files = st.file_uploader("Fehlerbilder hochladen", type=['jpg', 'png'], accept_multiple_files=True)

results = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

        with st.spinner('Analysiere Bild...'):
            class_idx, confidence = classify_image(image)
            fehler = CLASSES[class_idx]

        if view == "Kundenerklärung":
            st.subheader(f"Erkannter Fehler: {fehler['name']}")
            st.markdown(f"**Wahrscheinlichkeit:** {confidence*100:.1f}%")
            st.markdown(f"**Erklärung:** {fehler['behebung']}")
        else:
            st.subheader(f"Erkannter Fehler: {fehler['name']}")
            st.markdown(f"**Wahrscheinlichkeit:** {confidence*100:.1f}%")
            st.markdown(f"**Technische Ursache:** {fehler['ursache']}")
            st.markdown(f"**Empfehlung zur Behebung:** {fehler['behebung']}")

        results.append({
            'fehler': fehler['name'],
            'ursache': fehler['ursache'],
            'behebung': fehler['behebung']
        })
        st.markdown("---")

    if st.button("Analyse als PDF herunterladen"):
        pdf_bytes = generate_pdf(results)
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="analysebericht.pdf">PDF herunterladen</a>'
        st.markdown(href, unsafe_allow_html=True)

    st.success("Analyse abgeschlossen. Bilder wurden nicht gespeichert.")
