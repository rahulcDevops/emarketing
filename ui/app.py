"""
ui/app.py
LeadSentry: Inbound Email Reply & Lead Routing Console
"""

import os
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
PREDICT_URL = f"{API_BASE_URL}/predict"
HEALTH_URL = f"{API_BASE_URL}/health"
FEEDBACK_URL = f"{API_BASE_URL}/feedback"

ACTION_DISPATCH = {
    "INTERESTED_DEMO": (" HOT LEAD", "Instant Slack Alert + Route to AE"),
    "HARD_UNSUBSCRIBE": (" SUPPRESSION RISK", "Immediate Blacklist from Mailbox"),
    "OUT_OF_OFFICE": (" AUTO-SNOOZE", "Pause Sequence for 14 Days"),
    "NOT_INTERESTED": (" ARCHIVE", "Polite Pass, Mark Inactive"),
    "WRONG_PERSON": (" REFERRAL", "Update Contact Details in CRM"),
    "OBJECTION_PRICING": (" PRICING OBJECTION", "Send Custom ROI One-Pager"),
}

st.set_page_config(page_title="LeadSentry", page_icon="📬", layout="centered")

st.title("📬 LeadSentry: B2B Reply Intent Router")
st.caption("Sub-5ms NLP routing for cold outbound mailboxes. Protects domain health & escalates hot leads.")

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "available_classes" not in st.session_state:
    st.session_state.available_classes = []

with st.sidebar:
    st.header("Service Health")
    try:
        health_resp = requests.get(HEALTH_URL, timeout=2)
        if health_resp.status_code == 200:
            hdata = health_resp.json()
            st.success("Routing Engine: Online")
            st.metric("Model Version", hdata.get("model_version", "unknown"))
            st.metric("Artifact SHA-256", hdata.get("artifact_hash", "none"))
            st.session_state.available_classes = hdata.get("classes", [])
        else:
            st.warning("Service Degraded")
    except Exception:
        st.error("Engine Offline")

samples = {
    "Select an incoming reply...": "",
    "Hot Lead (Demo Request)": "Thanks for reaching out! Do you have time Thursday at 2 PM to demo the platform?",
    "CAN-SPAM Risk (Unsubscribe)": "STOP emailing me! Remove our company from your list immediately or I will report you.",
    "Out of Office": "Auto-Reply: I am out of the office on annual leave until next Monday with no email access.",
    "Budget Blocker": "We liked your demo video, but your pricing is way too high for our current runway.",
    "Wrong Contact": "I am no longer leading the infrastructure team. Reach out to Alex, our VP of Eng.",
}

selected = st.selectbox("Load sample inbound reply:", options=list(samples.keys()))
initial_text = samples[selected] if selected != "Select an incoming reply..." else ""

with st.form("triage_form"):
    email_text = st.text_area("Inbound Email Body", value=initial_text, height=140)
    submitted = st.form_submit_button("Route Inbound Reply", use_container_width=True)

if submitted:
    cleaned = email_text.strip()
    if len(cleaned) < 3:
        st.warning("Please enter at least 3 characters.")
    else:
        with st.spinner("Analyzing intent..."):
            try:
                res = requests.post(PREDICT_URL, json={"text": cleaned}, timeout=3)
                if res.status_code == 200:
                    st.session_state.last_prediction = res.json()
                    st.session_state.last_prediction["text"] = cleaned
                else:
                    st.error("Error from inference engine.")
            except Exception as e:
                st.error(f"Failed to connect: {e}")

if st.session_state.last_prediction:
    pred = st.session_state.last_prediction
    cat = pred["category"]
    badge, action = ACTION_DISPATCH.get(cat, ("UNKNOWN", "Manual Review"))

    st.markdown("---")
    st.subheader(f"Classified Intent: `{cat}`")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Priority Level", pred["priority"])
    with col2:
        st.metric("Automated Route", pred["action"])

    st.progress(pred["confidence"])
    st.caption(f"Confidence: {pred['confidence']:.2%} | Serving Model: {pred['model_version']}")

    st.markdown("#### Sales Ops Validation (Human-in-the-Loop)")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Confirm Classification", use_container_width=True):
            requests.post(
                FEEDBACK_URL,
                json={
                    "ticket_text": pred["text"],
                    "predicted_category": cat,
                    "corrected_category": cat,
                    "confidence": pred["confidence"],
                    "model_version": pred["model_version"],
                    "is_correct": True,
                },
            )
            st.success("Confirmed and logged.")
    with c2:
        with st.expander("Re-route Category"):
            opts = st.session_state.available_classes or [cat]
            corrected = st.selectbox("True Intent:", options=opts)
            if st.button("Submit Re-routing", use_container_width=True):
                requests.post(
                    FEEDBACK_URL,
                    json={
                        "ticket_text": pred["text"],
                        "predicted_category": cat,
                        "corrected_category": corrected,
                        "confidence": pred["confidence"],
                        "model_version": pred["model_version"],
                        "is_correct": False,
                    },
                )
                st.success(f"Re-routed to {corrected}.")
