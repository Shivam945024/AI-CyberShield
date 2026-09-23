import streamlit as st
import pandas as pd

from detectors.url_detector import URLDetector
from detectors.email_detector import EmailDetector
from detectors.network_detector import NetworkDetector


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI-CyberShield",
    page_icon="🛡️",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🛡️ AI-CyberShield")
st.subheader("AI-Powered Cybersecurity Threat Detection Platform")

st.markdown(
    """
    AI-CyberShield is a unified cybersecurity platform for detecting
    malicious URLs, phishing emails and suspicious network activity.
    """
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🔐 Detection Modules")

module = st.sidebar.selectbox(
    "Select Detection Module",
    [
        "🏠 Dashboard",
        "🌐 URL Detection",
        "📧 Email Detection",
        "🌐 Network Detection",
        "🧑 Face Detection",
        "🖼️ Image Deepfake Detection",
        "🎥 Video Deepfake Detection",
        "🎵 Audio Deepfake Detection"
    ]
)


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if module == "🏠 Dashboard":

    st.header("AI-CyberShield Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🌐 URL Detection",
            "Active"
        )

    with col2:
        st.metric(
            "📧 Email Detection",
            "Active"
        )

    with col3:
        st.metric(
            "🌐 Network Detection",
            "Active"
        )

    st.divider()

    st.info(
        "Select a detection module from the sidebar to analyze a threat."
    )

    st.markdown(
        """
        ### Current Modules

        - 🌐 Malicious URL Detection
        - 📧 Phishing Email Detection
        - 🌐 Network Threat Detection
        - 🧑 Face Deepfake Detection
        - 🖼️ Image Deepfake Detection
        - 🎥 Video Deepfake Detection
        - 🎵 Audio Deepfake Detection

        ### Security Pipeline

        `Input → Feature Extraction → AI Model → Risk Analysis → Result`
        """
    )


# --------------------------------------------------
# URL DETECTION
# --------------------------------------------------

elif module == "🌐 URL Detection":

    st.header("🌐 Malicious URL Detection")

    url = st.text_input(
        "Enter URL",
        placeholder="https://example.com"
    )

    if st.button("🔍 Analyze URL"):

        if not url.strip():

            st.warning("Please enter a URL.")

        else:

            try:

                detector = URLDetector()

                result = detector.predict(url)

                st.subheader("Detection Result")

                if isinstance(result, dict):

                    prediction = result.get(
                        "prediction",
                        result.get("label", "Unknown")
                    )

                    risk = result.get(
                        "risk",
                        result.get("risk_score", "N/A")
                    )

                    st.write(
                        f"**Prediction:** {prediction}"
                    )

                    st.write(
                        f"**Risk:** {risk}"
                    )

                    if str(prediction).lower() in [
                        "malicious",
                        "phishing",
                        "unsafe",
                        "suspicious"
                    ]:

                        st.error("⚠️ Potentially dangerous URL")

                    else:

                        st.success("✅ URL appears safe")

                else:

                    st.write(result)

            except Exception as e:

                st.error(
                    f"URL detector error: {str(e)}"
                )


# --------------------------------------------------
# EMAIL DETECTION
# --------------------------------------------------

elif module == "📧 Email Detection":

    st.header("📧 Phishing Email Detection")

    sender = st.text_input(
        "Sender Email",
        placeholder="example@gmail.com"
    )

    subject = st.text_input(
        "Email Subject"
    )

    body = st.text_area(
        "Email Content",
        height=250,
        placeholder="Paste the email content here..."
    )

    if st.button("🔍 Analyze Email"):

        if not body.strip():

            st.warning("Please enter email content.")

        else:

            try:

                detector = EmailDetector()

                result = detector.predict(
                    sender=sender,
                    subject=subject,
                    body=body
                )

                st.subheader("Detection Result")

                if isinstance(result, dict):

                    prediction = result.get(
                        "prediction",
                        result.get("label", "Unknown")
                    )

                    risk = result.get(
                        "risk",
                        result.get("risk_score", "N/A")
                    )

                    st.write(
                        f"**Prediction:** {prediction}"
                    )

                    st.write(
                        f"**Risk:** {risk}"
                    )

                    if str(prediction).lower() in [
                        "phishing",
                        "malicious",
                        "suspicious",
                        "spam"
                    ]:

                        st.error(
                            "⚠️ Potential phishing/malicious email"
                        )

                    else:

                        st.success(
                            "✅ Email appears legitimate"
                        )

                else:

                    st.write(result)

            except Exception as e:

                st.error(
                    f"Email detector error: {str(e)}"
                )


# --------------------------------------------------
# NETWORK DETECTION
# --------------------------------------------------

elif module == "🌐 Network Detection":

    st.header("🌐 Network Threat Detection")

    col1, col2 = st.columns(2)

    with col1:

        source_ip = st.text_input(
            "Source IP",
            placeholder="192.168.1.10"
        )

        destination_ip = st.text_input(
            "Destination IP",
            placeholder="8.8.8.8"
        )

        protocol = st.selectbox(
            "Protocol",
            [
                "TCP",
                "UDP",
                "ICMP",
                "HTTP",
                "HTTPS"
            ]
        )

    with col2:

        source_port = st.number_input(
            "Source Port",
            min_value=0,
            max_value=65535,
            value=443
        )

        destination_port = st.number_input(
            "Destination Port",
            min_value=0,
            max_value=65535,
            value=80
        )

        packet_size = st.number_input(
            "Packet Size",
            min_value=0,
            value=500
        )

    if st.button("🔍 Analyze Network Traffic"):

        try:

            detector = NetworkDetector()

            result = detector.predict(
                source_ip=source_ip,
                destination_ip=destination_ip,
                protocol=protocol,
                source_port=source_port,
                destination_port=destination_port,
                packet_size=packet_size
            )

            st.subheader("Detection Result")

            if isinstance(result, dict):

                prediction = result.get(
                    "prediction",
                    result.get("label", "Unknown")
                )

                risk = result.get(
                    "risk",
                    result.get("risk_score", "N/A")
                )

                st.write(
                    f"**Prediction:** {prediction}"
                )

                st.write(
                    f"**Risk:** {risk}"
                )

                if str(prediction).lower() in [
                    "malicious",
                    "attack",
                    "intrusion",
                    "suspicious"
                ]:

                    st.error(
                        "🚨 Suspicious network activity detected"
                    )

                else:

                    st.success(
                        "✅ Network traffic appears normal"
                    )

            else:

                st.write(result)

        except Exception as e:

            st.error(
                f"Network detector error: {str(e)}"
            )


# --------------------------------------------------
# FACE DETECTION
# --------------------------------------------------

elif module == "🧑 Face Detection":

    st.header("🧑 Face Deepfake Detection")

    st.info(
        "Face deepfake detection module will be integrated in Phase 2."
    )

    uploaded_file = st.file_uploader(
        "Upload Face Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        st.image(
            uploaded_file,
            caption="Uploaded Face",
            use_container_width=True
        )

        st.warning(
            "Model integration pending."
        )


# --------------------------------------------------
# IMAGE DEEPFAKE DETECTION
# --------------------------------------------------

elif module == "🖼️ Image Deepfake Detection":

    st.header("🖼️ Image Deepfake Detection")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        st.image(
            uploaded_file,
            caption="Uploaded Image",
            use_container_width=True
        )

        st.warning(
            "Deepfake image model integration pending."
        )


# --------------------------------------------------
# VIDEO DEEPFAKE DETECTION
# --------------------------------------------------

elif module == "🎥 Video Deepfake Detection":

    st.header("🎥 Video Deepfake Detection")

    uploaded_file = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_file:

        st.video(uploaded_file)

        st.warning(
            "Deepfake video model integration pending."
        )


# --------------------------------------------------
# AUDIO DEEPFAKE DETECTION
# --------------------------------------------------

elif module == "🎵 Audio Deepfake Detection":

    st.header("🎵 Audio Deepfake Detection")

    uploaded_file = st.file_uploader(
        "Upload Audio",
        type=["wav", "mp3", "ogg", "m4a"]
    )

    if uploaded_file:

        st.audio(uploaded_file)

        st.warning(
            "Deepfake audio model integration pending."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "🛡️ AI-CyberShield | AI-Based Cybersecurity & Deepfake Detection"
)
