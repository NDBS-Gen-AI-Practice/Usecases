
import streamlit as st
import ollama
import easyocr
from PIL import Image
import cv2
import numpy as np
import io

# Page configuration
st.set_page_config(
    page_title="Local OCR",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description in main area
st.title("🖼️ Local OCR")

# Add clear button to top right
col1, col2 = st.columns([6,1])
with col2:
    if st.button("Clear 🗑️"):
        if 'ocr_result' in st.session_state:
            del st.session_state['ocr_result']
        if 'easyocr_result' in st.session_state:
            del st.session_state['easyocr_result']
        st.rerun()

st.markdown('<p style="margin-top: -20px;">Extract structured text from images using Local OCR and EasyOCR!</p>', unsafe_allow_html=True)
st.markdown("---")

# Move upload controls to sidebar
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")
        
        if st.button("Extract Text 🔍", type="primary"):
            with st.spinner("Processing image..."):
                try:
                    # Local OCR using Llama
                    response = ollama.chat(
                        model='llama3.2-vision',
                        messages=[{
                            'role': 'user',
                            'content': """Analyze the text in the provided image. Extract all readable content
                                        and present it in a clear, plain text format.""",
                            'images': [uploaded_file.getvalue()]
                        }]
                    )
                    st.session_state['ocr_result'] = response.message.content
                except Exception as e:
                    st.error(f"Error processing image with Local OCR: {str(e)}")

                try:
                    # EasyOCR Extraction
                    image_bytes = uploaded_file.getvalue()
                    np_img = np.frombuffer(image_bytes, np.uint8)
                    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                    reader = easyocr.Reader(['en'])
                    result = reader.readtext(img)
                    extracted_text = "\n".join([text for _, text, _ in result])
                    st.session_state['easyocr_result'] = extracted_text
                except Exception as e:
                    st.error(f"Error processing image with EasyOCR: {str(e)}")

# Main content area for results
if 'ocr_result' in st.session_state or 'easyocr_result' in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        if 'ocr_result' in st.session_state:
            st.header("Local OCR Result")
            st.text(st.session_state['ocr_result'])
    with col2:
        if 'easyocr_result' in st.session_state:
            st.header("EasyOCR Result")
            st.text(st.session_state['easyocr_result'])
else:
    st.info("Upload an image and click 'Extract Text' to see the results here.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Local OCR and EasyOCR | [Report an Issue]")

