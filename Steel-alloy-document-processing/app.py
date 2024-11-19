from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
import os
import markdown
from xhtml2pdf import pisa
import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
import re
import json
import tempfile
import base64
import io
from io import BytesIO


# Azure Form Recognizer endpoint and API key
endpoint = os.getenv('ENDPOINT')
key = os.getenv('KEY')

# LLM Credentials
llm_key = os.getenv('GROQ_API_KEY')

# Initialize the DocumentAnalysisClient
credential = AzureKeyCredential(key)
document_analysis_client = DocumentAnalysisClient(endpoint, credential)

# Function to clean and normalize the text
def clean_text(text):
    text = re.sub(r'\s*\n\s*', '\n', text)  # Remove extra spaces around new lines
    text = re.sub(r'\s*:\s*', ': ', text)   # Ensure there's a space after colons
    text = re.sub(r'\s*->\s*', ' -> ', text)  # Ensure there's a space around arrows
    text = re.sub(r'\n+', '\n', text)  # Remove multiple consecutive newlines
    return text

def extract_markdown_content(markdown_text):
    """
    Extract content between ``` markers in a string.
    
    Parameters:
    markdown_text (str): The input string containing the markers.
    
    Returns:
    str: The extracted content in Markdown format.
    """
    pattern = re.compile(r'```(.*?)```', re.DOTALL)
    match = pattern.search(markdown_text)
    
    if match:
        return match.group(1).strip()
    else:
        return markdown_text
    
def extract_key_value_pairs(text):
    # Clean the text
    text = clean_text(text)
   
    key_value_pairs = {}
    lines = text.split('\n')
    current_key = None
    current_value = []

    for line in lines:
        line = line.strip()
        # Match key-value pairs with different delimiters and formats
        if re.match(r'.*:\s.*', line) or re.match(r'.*\s*->\s*.*', line):
            if current_key:
                key_value_pairs[current_key] = ' '.join(current_value).strip()
            if ':' in line:
                parts = line.split(':', 1)
            elif '->' in line:
                parts = line.split('->', 1)
            current_key = parts[0].strip()
            current_value = [parts[1].strip()]
        else:
            if current_key:
                current_value.append(line.strip())

    if current_key:
        key_value_pairs[current_key] = ' '.join(current_value).strip()

    return key_value_pairs

# Function to convert PDF file to base64 string
def pdf_to_base64(pdf_file):
    with open(pdf_file, "rb") as f:
        pdf_bytes = f.read()
    encoded_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    return f"data:application/pdf;base64,{encoded_pdf}"

# Streamlit UI
st.title("Steel Alloy Entities Extraction")

# Sidebar for logo and file uploader
st.sidebar.image("company_logo.png", use_column_width=True)
uploaded_file = st.sidebar.file_uploader("Uplpad Data document", type="pdf")
uploaded_template_file = st.sidebar.file_uploader("Upload template", type='pdf')

# Initialize llm
llm = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile")

if uploaded_file is not None:
    st.subheader("File Preview")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.markdown(f'<iframe src="{pdf_to_base64(pdf_path)}" width="600" height="700"></iframe>', unsafe_allow_html=True)

    if st.button("Extract Content"):
        with st.spinner("Processing..."):
            with open(pdf_path, "rb") as f:
                analyze_result = document_analysis_client.begin_analyze_document("prebuilt-layout", document=f).result()

            extracted_text = analyze_result.content
            modified_text = clean_text(extracted_text)

            # Extract key-value pairs from the OCR content
            key_value_pairs = extract_key_value_pairs(extracted_text)

            extracted_json_path = "extracted_key_value_pairs.json"
            with open(extracted_json_path, "w") as file:
                json.dump(key_value_pairs, file, indent=4)

            st.write("")  # Add space
            st.subheader("Original Content")
            st.text_area("Raw Content", value=extracted_text, height=400)

            st.subheader("Extracted Key-Value Pairs")
            st.code(json.dumps(key_value_pairs, indent=4), language='json', line_numbers=False)

            st.write("")  # Add space

            # Add CSS for horizontal scrollbar
            st.write(
                """
                <style>
                .stTextArea textarea {
                    white-space: nowrap;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

        # LLM to generate a Sales order.
        st.subheader("Technical Formula")
        with st.spinner("Generating Technical Formula..."):
            st.write("")
            if uploaded_template_file is not None:
                template_bytes = uploaded_template_file.read()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_template_file:
                    tmp_template_file.write(template_bytes)
                    template_path = tmp_template_file.name
                
                pdf_loader = PyPDFLoader(template_path)
                docs = pdf_loader.load()
                prompt_template = """
                Task: You are given a steel-alloy material specification document.
                You are an expert in determining the technical formula used in production line to create a product and also consume the knowledge related to making steel alloy products.
                You will create a document form the input text you are given to create a technical formula document that can be given to production line.
                Regarding the technical formula, it is the chemical composition, Mechanical properties and the Metallurgical Characteristics used on the composition of steel alloys. 
                Identify all the necessary components for the composition of steel alloys to make a technical formula professional document with all the needed names, dates, revisions, standards etc.
                Input: {template} {text}
                Output: A markdown formatted professional looking technical formula document. 
                Instructions: 
                Do not go out of the context and keep your response concise and precise. 
                Do not make up information that is not there in the document provided. 

                """
                prompt = PromptTemplate.from_template(prompt_template)
                llm_chain = LLMChain(llm=llm, prompt=prompt)
                result = llm_chain.invoke(input={'template': docs, 'text': modified_text})
                llm_output = result['text']
                st.markdown(llm_output)
            else:
                prompt_template = """
                Task: You are given a steel-alloy material specification document.
                You are an expert in determining the technical formula used in production line to create a product and also consume the knowledge related to making steel alloy products.
                You will create a document form the input text you are given to create a technical formula document that can be given to production line.
                Regarding the technical formula, it is the chemical composition, Mechanical properties used on the composition of steel alloys. 
                Identify all the necessary components for the composition of steel alloys to make a technical formula professional document with all the needed names, dates, revisions, standards etc.
                If the values are not mentioned in the Text then just say in that place "Not mentioned". 
                Follow the below template and produce it as output.
                Input: {text}
                Output: 
                Technical Formula Template:
                        ```
                        # **Technical Formula Document - Steel Alloy [Alloy Name]**
                        - **Date**: [Date]
                        - **Review/Revision**: (if applicable)
                        - **Reference/Standard**: [Reference or Standard Number]
                        - **Product**: (if applicable)
                        - **Supplier**: [Supplier Name] (if applicable)

                        ### **Chemical Composition**:

                        ### **Mechanical Properties**:

                        ## **Additional Notes**:
                        - [Any other relevant information]

                ```
            Instructions:
            Do not go out of the context and keep your response concise and precise. 
            Only provide the information provided in the document. 
            If the chemical composition is in a tabular format, please provide your response too in a tabular format.
            If there are blanks in chemical composition, then mentioned 'not mentioned'.
            Do not make up information that is not there in the document provided. 
            Follow the template above for the structure of the document.
            Do not put any unnecessary characters.
            Keep your responses precise, concise and clear.
            

"""
                prompt = PromptTemplate.from_template(prompt_template)
                llm_chain = LLMChain(llm=llm, prompt=prompt)
                result = llm_chain.invoke(modified_text)
                llm_output = result['text']
                st.markdown(llm_output)

        
        # Generating a Markdown File
        def generate_pdf_file(llm_output):
            if '```' in llm_output:
                content = extract_markdown_content(llm_output)
            else:
                content = llm_output

            # Convert Markdown to HTML
            html_content = markdown.markdown(content, extensions=['tables'])

            # Add basic HTML structure
            full_html = f"""
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: Arial, sans-serif; }}
                    table, th, td {{ border: 1px solid black; border-collapse: collapse; padding: 5px; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """

            # Convert HTML to PDF
            pdf_file = BytesIO()
            pisa_status = pisa.CreatePDF(
                src=full_html,
                dest=pdf_file
            )
            pdf_file.seek(0)

            if pisa_status.err:
                return None
            return pdf_file
        pdf_file = generate_pdf_file(llm_output)




        if pdf_file:
            st.download_button(
        label="Download Technical Formula as PDF",
        data=pdf_file,
        file_name="technical_formula.pdf",
        mime="application/pdf"
    )
        else:
            st.error("An error occurred while generating the PDF.")