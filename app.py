import streamlit as st
import pdfplumber
import fitz  # PyMuPDF
import json
import requests

st.set_page_config(page_title="AI Document Orchestrator", layout="wide")

N8N_WEBHOOK_URL = st.secrets["N8N_WEBHOOK_URL"]
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

def extract_text(file):
    if file.name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")

    else:
        return ""

def _extract_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        return json.loads(candidate)

    raise json.JSONDecodeError("No valid JSON found in model output", text, 0)


def gemini_dynamic_extraction(document_text, user_question):
    prompt = f"""
You are an AI system performing vendor onboarding and security risk assessment.

TASK:
Based on the USER QUESTION, identify and extract the 5-8 MOST RELEVANT key-value fields
from the document text that are REQUIRED to answer the question.

Rules:
- Fields must be dynamically chosen (not fixed).
- Values must come strictly from the document.
- If risk or approval is discussed, add "risk_level" (Low / Medium / High) and a short "risk_rationale".
- Preserve exact dates, amounts, SLA metrics, and retention windows as written.
- Output MUST be valid JSON only.

DOCUMENT:
{document_text}

USER QUESTION:
{user_question}
"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Return ONLY valid JSON. Do not include any extra text.",
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "max_tokens": 2048,
        },
        timeout=60,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]

    return _extract_json(content)

st.title("?? Vendor Risk Document Orchestrator")

uploaded_file = st.file_uploader("Upload Document (.pdf or .txt)", type=["pdf", "txt"])
question = st.text_input("Ask a vendor-risk question about the document")

structured_data = None
doc_text = None

if uploaded_file and question:
    with st.spinner("Extracting document text..."):
        doc_text = extract_text(uploaded_file)

    with st.spinner("Running Gemini Dynamic Extraction..."):
        structured_data = gemini_dynamic_extraction(doc_text, question)

    st.subheader("?? Structured Data Extracted (JSON)")
    st.json(structured_data)

if structured_data:
    st.subheader("?? Conditional Alert Automation")

    recipient_email = st.text_input("Recipient Email ID")

    if st.button("Send Alert Mail"):
        payload = {
            "document_text": doc_text,
            "question": question,
            "structured_data": structured_data,
            "recipient_email": recipient_email
        }

        with st.spinner("Triggering n8n workflow..."):
            response = requests.post(N8N_WEBHOOK_URL, json=payload)

        if response.status_code != 200:
            st.error(f"n8n error: {response.status_code} {response.text}")
            st.stop()

        try:
            result = response.json()
        except ValueError:
            st.error(f"n8n returned non-JSON response: {response.text}")
            st.stop()

        st.subheader("?? Final Analytical Answer")
        st.write(result.get("final_answer"))

        st.subheader("?? Generated Email Body")
        st.text_area("Email Body", result.get("email_body", "No email sent"), height=200, label_visibility="collapsed")

        st.subheader("? Email Automation Status")
        st.success(result.get("status"))
