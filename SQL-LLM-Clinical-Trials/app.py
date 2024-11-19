import gradio as gr
import time
from sqlalchemy import create_engine
import pandas as pd
import re
import duckdb
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_sql_query_chain
import os

# Load environment variables for secure API keys
load_dotenv()

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv('AZURE_OPENAI_ENDPOINT')

# Define valid dataset columns
VALID_COLUMNS = {
    'id_info', 'overall_status', 'study_type', 'eligibility_minimum_age',
    'location_countries', 'study_first_submitted', 'last_update_posted',
    'phase', 'intervention_type', 'intervention_name', 'completion_date',
    'enrollment', 'condition'
}

# Set up default login credentials
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "password"

# Create an SQLAlchemy engine for the DuckDB database
engine = create_engine('duckdb:///clinical_trial_structural_data.duckdb')
db = SQLDatabase(engine=engine)

# Define helper functions for SQL and response generation
def extract_table_columns(sql_query):
    column_pattern = r'\b\w+\.(\w+)\b|\b(\w+)\b'
    matches = re.findall(column_pattern, sql_query)
    
    columns = set()
    for match in matches:
        column = match[0] if match[0] else match[1]
        if column in VALID_COLUMNS:
            columns.add(column)
    
    return list(columns)

def generate_sql_code(prompt):
    llm = AzureChatOpenAI(
        openai_api_version="2023-06-01-preview",
        azure_deployment="gpt-4",
        temperature=0.01
    )

    answer_prompt = PromptTemplate.from_template(
        """You are an AI assistant specialized in answering questions about clinical trials based on database queries.
          Given the following user question, corresponding SQL query, and SQL result, provide a clear and well-structured answer.
        Instructions:
        - Format the answer in a very crisp manner in points.
        - Ensure that the answer is precise, relevant, and concise.
        - If the data contains repeating values, avoid redundancy in your response.
        - If the question is irrelevant to the data, respond with "I am unsure" and do not generate SQL code, References.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}

    Answer: """
    )
    
    execute_query = QuerySQLDataBaseTool(db=db)
    write_query = create_sql_query_chain(llm, db)
    chain = write_query | execute_query
    answer = answer_prompt | llm | StrOutputParser()
    
    chain = (
        RunnablePassthrough.assign(query=write_query).assign(
            result=itemgetter("query") | execute_query
        )
        | answer
    )
    
    sql_chain = create_sql_query_chain(llm, db=db)
    sql_code = sql_chain.invoke({"question": prompt})
    response = chain.invoke({"question": prompt})
    
    return sql_code, response

def process_prompt(prompt):
    """Processes the user prompt and returns SQL code, response, and processing time."""
    start_time = time.time()
    sql_code, response = generate_sql_code(prompt)
    end_time = time.time()
    processing_time = f"{end_time - start_time:.2f} seconds"

    # Extract and verify columns
    db_columns = extract_table_columns(sql_code)
    columns_extracted = f"Columns extracted: {', '.join(db_columns)}" if db_columns else "No columns extracted from query."

    return sql_code, response, processing_time, columns_extracted

# Login function to validate credentials
def login(username, password):
    if username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True, value="Invalid credentials, please try again.")

# Define Gradio UI layout with login and main app interfaces
with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("IBM Plex Sans"), "Arial", "sans-serif"])) as demo:
    # Login page
    with gr.Column(visible=True) as login_page:
        gr.Markdown("## 🔒 Please Log In")
        username_input = gr.Textbox(label="Username")
        password_input = gr.Textbox(label="Password", type="password")
        login_button = gr.Button("Login")
        login_message = gr.Textbox(visible=False, interactive=False)

    # Main application page
    with gr.Column(visible=False) as main_app:
        gr.Markdown("## 🤖 Enhanced Clinical Trials Querying")

        with gr.Row():
            prompt_input = gr.Textbox(placeholder="Enter your prompt here...", lines=3, label="Prompt")
            submit_btn = gr.Button("Execute")

        with gr.Row():
            sql_code_output = gr.Textbox(label="Generated SQL Code", lines=5, interactive=False)
            response_output = gr.Textbox(label="Generated Response", lines=5, interactive=False)

        with gr.Row():
            processing_time_output = gr.Textbox(label="Processing Time", interactive=False)
            columns_output = gr.Textbox(label="References", interactive=False)

        # Update outputs on submit button click
        submit_btn.click(
            fn=process_prompt,
            inputs=prompt_input,
            outputs=[sql_code_output, response_output, processing_time_output, columns_output]
        )

    # Show the main app only if login is successful
    login_button.click(
        login, 
        inputs=[username_input, password_input], 
        outputs=[main_app, login_message]
    )

if __name__ == "__main__":
    # Get the port from the environment variable, default to 8080
    port = int(os.environ.get('PORT', '8080'))
    print(f"Starting app on port {port} and server_name 0.0.0.0")
    try:
        # Launch the app with server_name and server_port
        demo.launch(server_name="0.0.0.0", server_port=port)
    except Exception as e:
        print(f"An error occurred: {e}")
