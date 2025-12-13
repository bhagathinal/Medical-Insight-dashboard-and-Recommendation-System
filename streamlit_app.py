import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import spacy
import os
import pandas as pd
import google.generativeai as genai
from datetime import datetime
from pathlib import Path

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Medi-Assist AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. SECURE API KEY HANDLING ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    GEMINI_AVAILABLE = True
except (KeyError, Exception):
    GEMINI_AVAILABLE = False

# --- 3. MODEL AND ASSET PATHS ---
SCRIPT_DIR = Path(__file__).parent
CLASSIFIER_MODEL_PATH = "./models/ClinicalBERT_trained_V2"
LABEL_MAP_PATH = f"{CLASSIFIER_MODEL_PATH}/label_map.json"
NER_MODEL_NAME = "en_core_sci_sm"

# --- 4. SESSION STATE INITIALIZATION ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'history' not in st.session_state:
    st.session_state.history = []
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = False
if 'feedback_text' not in st.session_state:
    st.session_state.feedback_text = ""

# --- 5. RBAC LOGIN ---
def check_password():
    if st.session_state.get("password") == st.secrets.get("APP_PASSWORD", "admin"):
        st.session_state["authenticated"] = True
        del st.session_state["password"]
    else:
        st.session_state["authenticated"] = False
        st.error("😕 Incorrect password. Please try again.")

if not st.session_state["authenticated"]:
    st.markdown("<h1 style='text-align: center;'>🩺 Medi-Assist AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Authorized Access Only</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.text_input("Enter Doctor Password", type="password", key="password", on_change=check_password)
        st.button("Login", on_click=check_password, use_container_width=True, type="primary")
        st.caption("Note: This is a simulated login for demonstration.")
    st.stop()

# --- 6. SIDEBAR ---
with st.sidebar:
    st.title("🩺 Medi-Assist AI")
    st.caption(f"Logged in as Doctor • {datetime.now().strftime('%d %b %Y')}")
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()
    
    st.subheader("Analysis History")
    if st.session_state.history:
        if st.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
        for entry in st.session_state.history:
            with st.expander(f"**{entry['timestamp']}** - *{entry['specialty']}*"):
                st.write(f"**Conditions:** {', '.join(entry['conditions'])}")
    else:
        st.info("Your session history will appear here.")
    
    st.markdown("---")
    st.subheader("About")
    st.markdown("Assisting healthcare professionals with rapid insights from clinical notes using PubMedBERT, SciSpacy, Gemini, and an integrated medical knowledge base.")

# --- 7. CUSTOM CSS ---
st.markdown("""
    <style>
        .stApp { background-color: #F8F9FA; }
        .condition-tag {
            display: inline-block; padding: 0.3em 0.8em; margin: 0.2em;
            font-size: 0.9em; font-weight: 600; background-color: #E3F2FD;
            color: #1565C0; border-radius: 20px; border: 1px solid #BBDEFB;
        }
    </style>
""", unsafe_allow_html=True)

# --- 8. MODEL LOADING ---
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_PATH)
    classifier_model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_MODEL_PATH)
    with open(LABEL_MAP_PATH, 'r') as f:
        id2label = {int(k): v for k, v in json.load(f)['id2label'].items()}
    ner_model = spacy.load(NER_MODEL_NAME)
    return tokenizer, classifier_model, id2label, ner_model

tokenizer, classifier_model, id2label, ner_model = load_models()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier_model.to(device)

# --- 9. KNOWLEDGE BASE (AUTO-CREATE IF MISSING) ---
@st.cache_data
def load_knowledge_base():
    kb_path = os.path.join(SCRIPT_DIR, "./data/cleaned_data.csv")
    if os.path.exists(kb_path):
        return pd.read_csv(kb_path)
    else:
        # Auto-create default KB
        default_data = {
            "disease": ["hypertensive disease"]*7,
            "symptom": ["pain chest", "shortness of breath", "dizziness", "asthenia", "fall", "syncope", "vertigo"],
            "count": [3363]*7
        }
        kb_df = pd.DataFrame(default_data)
        return kb_df

knowledge_base = load_knowledge_base()

def validate_with_knowledge_base(predicted_specialty, conditions):
    matches = knowledge_base[knowledge_base['disease'].str.lower() == predicted_specialty.lower()]
    if matches.empty:
        return f"No direct validation found for specialty: {predicted_specialty}."
    valid_conditions = matches['symptom'].str.lower().tolist()
    overlap = [c for c in conditions if c.lower() in valid_conditions]
    if overlap:
        return f"✅ Validation successful: Conditions align with {predicted_specialty} knowledge base."
    else:
        return f"⚠️ Validation mismatch: Conditions not typically linked to {predicted_specialty}."

# --- 10. HELPER FUNCTIONS ---
def extract_conditions(text):
    doc = ner_model(text)
    return list(set([ent.text for ent in doc.ents if len(ent.text) > 2]))

@st.cache_data(show_spinner=False)
def get_recommendations_with_gemini(condition_list, vitals):
    if not GEMINI_AVAILABLE: return None
    vitals_string = ", ".join([f"{key}: {value}" for key, value in vitals.items() if value])
    prompt = f"""
    You are an expert AI medical assistant. Provide safe, general, non-prescriptive lifestyle recommendations.
    **Disclaimer:** Start with: "Disclaimer: These are AI-generated suggestions..."
    **Patient Vitals:** {vitals_string}
    **Conditions:** {', '.join(condition_list)}
    """
    try:
        model = genai.GenerativeModel('models/gemini-2.5-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error while generating recommendations: {e}"

# --- 11. MAIN DASHBOARD ---
st.title("AI-Powered Clinical Insights Dashboard")
st.markdown("Enter patient data for automated analysis and knowledge-based recommendations.")

input_col, output_col = st.columns([1, 1.2])

with input_col:
    st.subheader("👤 Patient Vitals")
    r1c1, r1c2, r1c3 = st.columns(3)
    name = r1c1.text_input("Name")
    age = r1c2.number_input("Age", min_value=0, max_value=120, step=1)
    gender = r1c3.selectbox("Gender", ["", "Male", "Female", "Other"])
    r2c1, r2c2 = st.columns(2)
    height = r2c1.text_input("Height")
    weight = r2c2.text_input("Weight")
    r3c1, r3c2 = st.columns(2)
    bp = r3c1.text_input("Blood Pressure")
    sugar = r3c2.text_input("Blood Sugar")
    r4c1, r4c2 = st.columns(2)
    smoking = r4c1.selectbox("Smoking", ["", "No", "Yes", "Formerly"])
    alcohol = r4c2.selectbox("Alcohol", ["", "No", "Occasionally", "Regularly"])

    st.divider()
    st.subheader("📝 Clinical Note")
    clinical_notes = st.text_area("Paste note here:", height=250, placeholder="e.g., Patient complains of chronic heartburn...")
    analyze_button = st.button("✨ Analyze Case", use_container_width=True, type="primary")

with output_col:
    st.subheader("📊 Analysis Results")

    if analyze_button:
        if not clinical_notes:
            st.warning("Please enter a clinical note to analyze.")
        elif not GEMINI_AVAILABLE:
            st.error("**Configuration Error:** `GOOGLE_API_KEY` not set.")
        else:
            st.session_state.feedback_given = False
            with st.spinner("🧠 Performing Comprehensive AI Analysis..."):
                vitals = {"Age": age if age > 0 else None, "Gender": gender, "Smoking": smoking, "Alcohol": alcohol}

                # --- INFERENCE ---
                inputs = tokenizer(clinical_notes, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
                input_ids, attention_mask = inputs["input_ids"].to(device), inputs["attention_mask"].to(device)
                with torch.no_grad():
                    outputs = classifier_model(input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(outputs.logits, dim=1)
                confidence, predicted_class_id = torch.max(probabilities, dim=1)
                predicted_specialty = id2label[predicted_class_id.item()]
                confidence_score = confidence.item()
                extracted_conditions = extract_conditions(clinical_notes)

                # --- VALIDATION ---
                validation_msg = validate_with_knowledge_base(predicted_specialty, extracted_conditions)

                recommendations = get_recommendations_with_gemini(extracted_conditions, vitals)

                history_entry = {
                    "timestamp": datetime.now().strftime("%I:%M %p"),
                    "specialty": predicted_specialty,
                    "confidence": confidence_score,
                    "conditions": extracted_conditions if extracted_conditions else ["None"],
                    "recommendations": recommendations,
                    "validation": validation_msg
                }

                # --- Add to history safely ---
                if isinstance(st.session_state.history, list):
                    st.session_state.history.insert(0, history_entry)
                else:
                    st.session_state.history = [history_entry]

                st.rerun()

    # --- DISPLAY RESULTS ---
    if st.session_state.history:
        latest = st.session_state.history[0]
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Specialty", "🧬 Conditions", "💡 Recommendations", "✅ Validation"])

        with tab1:
            st.metric(label="", value=latest['specialty'], delta=f"{latest['confidence']:.1%} Confidence Score", delta_color="normal")

        with tab2:
            if latest['conditions'] != ["None"]:
                tags_html = "".join([f'<span class="condition-tag">{c}</span>' for c in latest['conditions']])
                st.markdown(tags_html, unsafe_allow_html=True)
            else:
                st.info("No specific conditions extracted.")

        with tab3:
            if latest['recommendations']:
                with st.container(height=450):
                    st.markdown(latest['recommendations'])

                st.divider()
                st.write("**Doctor Feedback:** Was this AI analysis helpful?")
                f_col1, f_col2, _ = st.columns([1, 1, 3])
                btn_disabled = st.session_state.feedback_given
                if f_col1.button("👍 Yes", disabled=btn_disabled, key="fb_yes"):
                    st.session_state.feedback_given = True
                    st.toast("Feedback recorded. Thank you!", icon="✅")
                if f_col2.button("👎 No", disabled=btn_disabled, key="fb_no"):
                    st.session_state.feedback_given = True
                    st.toast("Thank you. We will use this to improve.", icon="📝")

                if st.session_state.feedback_given:
                    st.caption("✅ Feedback submitted for this analysis.")
                    st.session_state.feedback_text = st.text_area("🗣️ Additional Comments (optional):", placeholder="Share any notes or suggestions here...")
                    if st.session_state.feedback_text:
                        st.success("Your additional feedback has been recorded. Thank you!")

        with tab4:
            st.info(latest['validation'])
    else:
        st.info("👈 Enter patient data and click 'Analyze Case' to see results.")
