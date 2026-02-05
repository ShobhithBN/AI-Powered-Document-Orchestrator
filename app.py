import streamlit as st
import pdfplumber
import json
import requests

st.set_page_config(page_title="AI Document Orchestrator", layout="wide")

N8N_WEBHOOK_URL = st.secrets.get("N8N_WEBHOOK_URL")
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

def _load_model_options():
    models = st.secrets.get("OPENROUTER_MODELS")
    if isinstance(models, str):
        models = [m.strip() for m in models.split(",") if m.strip()]
    if isinstance(models, (list, tuple)):
        cleaned = []
        seen = set()
        for m in models:
            name = str(m).strip()
            if name and name not in seen:
                cleaned.append(name)
                seen.add(name)
        return cleaned
    return []

@st.cache_data(ttl=3600)
def _fetch_openrouter_models():
    response = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    return data if isinstance(data, list) else []


def _get_free_model_options():
    try:
        models = _fetch_openrouter_models()
    except Exception as exc:
        return [], str(exc)

    def _is_zero(value):
        try:
            return float(value) == 0.0
        except (TypeError, ValueError):
            return False

    free_models = []
    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        if not model_id:
            continue
        if isinstance(model_id, str) and model_id.endswith(":free"):
            free_models.append(model_id)
            continue
        pricing = model.get("pricing") or {}
        if _is_zero(pricing.get("prompt")) and _is_zero(pricing.get("completion")) and _is_zero(
            pricing.get("request")
        ):
            free_models.append(model_id)

    free_models = sorted(set(free_models))
    if "openrouter/free" not in free_models:
        free_models.insert(0, "openrouter/free")
    else:
        free_models = ["openrouter/free"] + [m for m in free_models if m != "openrouter/free"]
    return free_models, None

def _get_common_model_options():
    return [
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "openai/gpt-4.1-mini",
        "openai/gpt-4.1",
        "openai/gpt-4.1-nano",
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3.5-haiku",
        "anthropic/claude-3-opus",
        "google/gemini-2.5-pro",
        "google/gemini-2.5-flash",
        "meta-llama/llama-3.1-70b-instruct",
        "meta-llama/llama-3.1-8b-instruct",
        "mistralai/mistral-large",
        "mistralai/mistral-small",
        "qwen/qwen-2.5-72b-instruct",
        "qwen/qwen-2.5-7b-instruct",
        "deepseek/deepseek-chat",
        "deepseek/deepseek-coder",
        "nousresearch/hermes-3-llama-3.1-70b",
    ]

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


def gemini_dynamic_extraction(document_text, user_question, model_name):
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
            "model": model_name,
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

def _get_key_ci(data, key):
    if not isinstance(data, dict):
        return None
    if key in data:
        return data[key]
    key_lc = key.lower()
    for k, v in data.items():
        if isinstance(k, str) and k.strip().lower() == key_lc:
            return v
    return None


def _normalize_n8n_response(result):
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            return {}

    def _has_keys(d):
        return (
            _get_key_ci(d, "final_answer") is not None
            or _get_key_ci(d, "email_body") is not None
            or _get_key_ci(d, "status") is not None
        )

    def _walk(obj):
        if isinstance(obj, dict):
            if _has_keys(obj):
                return obj
            if isinstance(obj.get("json"), dict):
                return obj["json"]
            for key in ("body", "data", "result"):
                if key in obj:
                    found = _walk(obj[key])
                    if found:
                        return found
            for value in obj.values():
                found = _walk(value)
                if found:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = _walk(item)
                if found:
                    return found
        return {}

    return _walk(result)

st.title("?? Vendor Risk Document Orchestrator")

free_model_options, free_models_error = _get_free_model_options()
common_model_options = _get_common_model_options()
model_options = _load_model_options()
with st.sidebar:
    st.subheader("Model Settings")
    model_source = st.selectbox(
        "Model List",
        ["Free (Live)", "Common", "Configured", "Manual"],
        index=0,
    )

    if model_source == "Free (Live)":
        if free_model_options:
            default_index = 0
            if OPENROUTER_MODEL in free_model_options:
                default_index = free_model_options.index(OPENROUTER_MODEL)
            selected_model = st.selectbox("OpenRouter Free Models", free_model_options, index=default_index)
        else:
            if free_models_error:
                st.caption("Free models list unavailable. Enter a model manually.")
            selected_model = st.text_input("OpenRouter Model", value=OPENROUTER_MODEL)
    elif model_source == "Common":
        default_index = 0
        if OPENROUTER_MODEL in common_model_options:
            default_index = common_model_options.index(OPENROUTER_MODEL)
        selected_model = st.selectbox("OpenRouter Common Models", common_model_options, index=default_index)
    elif model_source == "Configured":
        if model_options:
            default_index = 0
            if OPENROUTER_MODEL in model_options:
                default_index = model_options.index(OPENROUTER_MODEL)
            selected_model = st.selectbox("OpenRouter Models", model_options, index=default_index)
        else:
            st.caption("No configured model list found. Enter a model manually.")
            selected_model = st.text_input("OpenRouter Model", value=OPENROUTER_MODEL)
    else:
        selected_model = st.text_input("OpenRouter Model", value=OPENROUTER_MODEL)

uploaded_file = st.file_uploader("Upload Document (.pdf or .txt)", type=["pdf", "txt"])
question = st.text_input("Ask a vendor-risk question about the document")

structured_data = None
doc_text = None

if uploaded_file and question:
    if not OPENROUTER_API_KEY:
        st.error("Missing OPENROUTER_API_KEY in secrets. Add it in .streamlit/secrets.toml.")
        st.stop()
    with st.spinner("Extracting document text..."):
        doc_text = extract_text(uploaded_file)

    with st.spinner("Running Gemini Dynamic Extraction..."):
        structured_data = gemini_dynamic_extraction(doc_text, question, selected_model)

    st.subheader("?? Structured Data Extracted (JSON)")
    st.json(structured_data)

if structured_data:
    st.subheader("?? Conditional Alert Automation")

    recipient_email = st.text_input("Recipient Email ID")

    if st.button("Send Alert Mail"):
        if not N8N_WEBHOOK_URL:
            st.error("Missing N8N_WEBHOOK_URL in secrets. Add it in .streamlit/secrets.toml.")
            st.stop()
        payload = {
            "document_text": doc_text,
            "question": question,
            "structured_data": structured_data,
            "recipient_email": recipient_email
        }

        with st.spinner("Triggering n8n workflow..."):
            try:
                response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=30)
            except requests.exceptions.RequestException as exc:
                st.error(f"Failed to reach n8n webhook: {exc}")
                st.info("If you're on Streamlit Cloud, a localhost URL won't work. Use a public n8n URL.")
                st.stop()

        if response.status_code != 200:
            st.error(f"n8n error: {response.status_code} {response.text}")
            st.stop()

        try:
            result = response.json()
        except ValueError:
            st.error(f"n8n returned non-JSON response: {response.text}")
            st.stop()

        result_data = _normalize_n8n_response(result)

        st.subheader("?? Final Analytical Answer")
        st.write(_get_key_ci(result_data, "final_answer"))

        st.subheader("?? Generated Email Body")
        st.text_area(
            "Email Body",
            _get_key_ci(result_data, "email_body") or "No email sent",
            height=200,
            label_visibility="collapsed",
        )

        st.subheader("? Email Automation Status")
        st.success(_get_key_ci(result_data, "status") or "Status not returned from n8n")
