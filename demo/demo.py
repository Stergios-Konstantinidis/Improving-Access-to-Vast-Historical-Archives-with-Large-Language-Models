import streamlit as st
import requests
import zipfile
import io
import os
import LayoutParser as lp
import pytesseract
st.title("Demo Application")
st.write("This is a simple demo application using Streamlit.")

if 'dataset' not in st.session_state:
    st.session_state.dataset = None
global images 
images = []

if 'lp_done' not in st.session_state:
    st.session_state.lp_done = False

if not st.session_state.dataset:
    if st.button("Test with ICDAR2023 dataset", help="use the dataset from ICDAR2023 for testing (https://ds4sd-icdar23-doclaynet-competition.s3.eu-de.cloud-object-storage.appdomain.cloud/competition-dataset-public.zip)"):
        st.write("Loading dataset...")
        response = requests.get("https://ds4sd-icdar23-doclaynet-competition.s3.eu-de.cloud-object-storage.appdomain.cloud/competition-dataset-public.zip")
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall("icdar2023_dataset")
            st.write("Dataset loaded successfully.")
            st.session_state.dataset = "/icdar2023_dataset"
        else:
            st.write("Failed to load dataset.")

    if st.button("Test with custom files", help="upload your own files for testing (files need to be in .png formats)"):
        uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
        if uploaded_files:
            os.makedirs("custom_files", exist_ok=True)
            for uploaded_file in uploaded_files:
                with open(os.path.join("custom_files", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.write(f"Uploaded {len(uploaded_files)} files successfully.")
            st.session_statedataset = "custom_files"
             
if st.session_state.dataset:
    st.write(f"Processing dataset: {st.session_state.dataset}")
    #load the dataset/PNG files and process with Layout Parser
    doc_files = []
    layout_parser_df, images = lp.generate_entries(directory=st.session_state.dataset + "/PNG/")
    if images != []:
        st.write("Processed Images:")
        cols = st.columns(5)
        for idx, img in enumerate(images):
            cols[idx % 5].image(img, use_container_width=True)
        st.session_state.lp_done = True

if st.session_state.lp_done:
    st.button("Run OCR on Detected Text Regions", help="This will run OCR on the detected text regions using Tesseract OCR (make sure Tesseract is installed on your system).")
    ocr_results = []
    for img in images:
        ocr_text = pytesseract.image_to_string(img)
        ocr_results.append(ocr_text)
        #add the ocr results to the cols as a caption
    st.write("OCR Results:")
    for idx, text in enumerate(ocr_results):
        st.write(f"Image {idx + 1} OCR Text:")
        st.text_area("", text, height=200)
        

            
