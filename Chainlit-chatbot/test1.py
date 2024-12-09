from io import BytesIO
import chainlit as cl
from pathlib import Path
import os
from vectors import EmbeddingsManager
from chatbot import ChatbotManager
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.docx import partition_docx
from langsmith import Client, traceable
from groq import Groq
import pymupdf
from gtts import gTTS
import tempfile
from Summarizer import summarizer
from Visualizer import visualizer
from AgentBuilder import agentBuilder
from GoalGenerator import goals_generate
from dotenv import load_dotenv
import base64
import json
import pandas as pd
import sqlite3
from operator import itemgetter
from chainlit.types import ThreadDict
import uuid
 
load_dotenv('.env')
groq_client = Groq()
os.environ['CHAINLIT_AUTH_SECRET'] = os.getenv("CHAINLIT_AUTH_SECRET")
os.environ['LITERAL_API_KEY'] = os.getenv("LITERAL_API_KEY")

import sqlite3

# Initialize the database and create the table
def initialize_chat_db():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            session_id TEXT,
            Userinput TEXT,
            response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


# Save a chat message to the database
def save_chat_message(session_id, userinput, response):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO chat_history (session_id, userinput, response)
        VALUES (?, ?, ?)
    """, (session_id, userinput, response))
    conn.commit()
    conn.close()

# Clear chat history for a session
def clear_chat_history(session_id):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        DELETE FROM chat_history WHERE session_id = ?
    """, (session_id,))
    conn.commit()
    conn.close()


welcome_message = """Welcome to the Chat with PDF, PPTX, DOCX, CSV or Excel To get started:
1. Upload a PDF, PPTX, DOCX file, CSV or Excel .
2. Ask a question about the file.
"""
 
def extract_text_from_pdf(pdf_path):
    document = pymupdf.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text
 
def extract_text_from_pptx(pptx_path):
    elements = partition_pptx(filename=pptx_path, strategy="HI_RES")
    text = "\n".join([str(element) for element in elements])
    return text
 
def extract_text_from_docx(docx_path):
    elements = partition_docx(filename=docx_path)
    text = "\n".join([str(element) for element in elements])
    return text

# Function to extract data from CSV or Excel files
def extract_data_from_file(file_path, delimiter=","):
    file_extension = file_path.split('.')[-1].lower()
    try:
        if file_extension == 'csv':
            df = pd.read_csv(file_path, delimiter=delimiter)
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")
        return df
    except Exception as e:
        raise ValueError(f"Error reading the file: {e}")
    
def authenticate_user(username: str, password: str) -> bool:
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()

    if result:
        stored_password = result[0]
        return password == stored_password
    return False


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if authenticate_user(username, password):
        return cl.User(
            identifier=username, metadata={"role": "user", "provider": "credentials"}
        )
    else:
        return None

def transcribe_audio(file, mime_type):
    try:
        transcription = groq_client.audio.transcriptions.create(
            file=(file.name, file),
            model="distil-whisper-large-v3-en",
            prompt="Specify context or spelling",
            response_format="json",
            language="en",
            temperature=0.0
        )
        return transcription.text
    except Exception as e:
        return f"Error in transcription: {e}"
 
@cl.on_chat_start
async def on_chat_start():
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("chat_history", [])

    initialize_chat_db()

    cl.user_session.set("chat_history", [])
    cl.user_session.set('tmp_dir', None)
    cl.user_session.set('chatbot_manager', None)
    cl.user_session.set('messages', [])
    cl.user_session.set("chat_history", [])
    cl.user_session.set("dataframe", None)
    cl.user_session.set("summary", None)
    cl.user_session.set("goals", None)
    cl.user_session.set("graph", None)

    file = None
    while file is None:
        file = await cl.AskFileMessage(
            content=welcome_message,
            accept=["application/pdf",
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "text/csv",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  
            ],
            max_size_mb=50,
        ).send()

    if not file or len(file) == 0:
        await cl.Message(content="No file was uploaded. Please try again.").send()
        return

    file = file[0]
    await cl.Message(content=f"Processing `{file.name}`...").send()

    tmp_dir = "tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    file_path = os.path.join(tmp_dir, file.name)
    with open(file.path, "rb") as source_file:
        with open(file_path, "wb") as destination_file:
            destination_file.write(source_file.read())

    cl.user_session.set("uploaded_file_path", file_path)

    try:
        if file.name.endswith(".pdf"):
            doc = extract_text_from_pdf(file_path)
        elif file.name.endswith(".pptx"):
            doc = extract_text_from_pptx(file_path)
        elif file.name.endswith(".docx"):
            doc = extract_text_from_docx(file_path)
        elif file.name.endswith((".csv", ".xls", ".xlsx")):
            delimiter = cl.user_session.get("delimiter", ",")
            df = extract_data_from_file(file_path, delimiter)
            cl.user_session.set("dataframe", df)
            preview = df.head().to_string(index=False)
            await cl.Message(content=f"File uploaded successfully! Here's a preview:\n\n```\n{preview}\n```").send()

            # Offer summary and goals generation as options
            actions = [
                cl.Action(name="generate_summary", value="generate_summary", description="Generate Summary"),
                cl.Action(name="ask_question", value="ask_question", description="Ask a Question")
            ]
            await cl.Message(content="You can now choose an action: Generate Summary, Generate Goals, or Ask a Question.", actions=actions).send()
            return

        # Process embeddings and initialize ChatbotManager for non-CSV files
        if file.name.endswith((".pdf", ".pptx", ".docx")):
            embeddings_manager = EmbeddingsManager(
                model_name="BAAI/bge-small-en",
                device="cpu",
                encode_kwargs={"normalize_embeddings": True},
                qdrant_url="https://72efe8dd-cc4e-4468-9047-a6d9de74edb2.us-east4-0.gcp.cloud.qdrant.io:6333",
                collection_name="vector_db_md"
            )
            embeddings_manager.create_embeddings(file_path)
            cl.user_session.set("embeddings_generated", True)

            chatbot_manager = ChatbotManager(
                model_name="BAAI/bge-small-en",
                device="cpu",
                encode_kwargs={"normalize_embeddings": True},
                llm_model="llama3-70b-8192",
                qdrant_url="https://72efe8dd-cc4e-4468-9047-a6d9de74edb2.us-east4-0.gcp.cloud.qdrant.io:6333",
                collection_name="vector_db_md"
            )
            cl.user_session.set("chatbot_manager", chatbot_manager)
            print("Chatbot Manager Initialized Successfully")
            await cl.Message(content=f"Successfully processed `{file.name}`. You can now ask questions.").send()
            return

    except Exception as e:
        await cl.Message(content=f"Failed to process the file: {e}").send()
        return
    
@cl.action_callback("generate_summary")
async def generate_summary_action(action: cl.Action):
    df = cl.user_session.get("dataframe")
    if df is None:
        await cl.Message(content="Please upload a file first.").send()
        return
    try:
        summary_result = summarizer(df)
        if isinstance(summary_result, dict):
            cl.user_session.set("summary", summary_result)
            await cl.Message(content="Summary generated successfully!").send()
            await cl.Message(content=json.dumps(summary_result, indent=2)).send()
            actions = [
                cl.Action(name="generate_goals", value="generate_goals", description="Generate Goals"),
            ]
            await cl.Message(content="You can now choose an action: Generate Goals or Ask a Question.", actions=actions).send()
        else:
            raise ValueError("Generated summary is not in a valid format.")
    except Exception as e:
        await cl.Message(content=f"Error generating summary: {e}").send()

    cl.user_session.set("graph", None)  # Clear any existing graphs
    await cl.Message(content="You can now ask a question about the dataset or request a visualization.").send()

@cl.action_callback("generate_goals")
async def generate_goals_action(action: cl.Action):
    summary = cl.user_session.get("summary")
    if summary is None:
        await cl.Message(content="Please generate a summary first.").send()
        return
    try:
        goals = goals_generate(summary)
        if isinstance(goals, list):
            cl.user_session.set("goals", goals)
            await cl.Message(content="Goals generated successfully!").send()
            await cl.Message(content=json.dumps(goals, indent=2)).send()
        else:
            raise ValueError("Generated goals are not in a valid format.")
    except Exception as e:
        await cl.Message(content=f"Error generating goals: {e}").send()

@cl.action_callback("ask_question")
async def ask_question_action(action: cl.Action):
    df = cl.user_session.get("dataframe")
    if df is None:
        await cl.Message(content="Please upload a dataset first.").send()
        return

    cl.user_session.set("graph", None)  # Clear any existing graphs
    await cl.Message(content="You can now ask a question about the dataset or request a visualization.").send()





def fetch_chat_history(session_id):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT userinput, response, timestamp FROM chat_history
        WHERE session_id = ?
        ORDER BY timestamp ASC
    """, (session_id,))
    history = cursor.fetchall()
    conn.close()
    return history

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    # Ensure session ID exists
    session_id = cl.user_session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        cl.user_session.set("session_id", session_id)

    # Fetch and replay chat history from the database
    chat_history = fetch_chat_history(session_id)
    if chat_history:
        for userinput, response, timestamp in chat_history:
            await cl.Message(content=response, author=userinput).send()

    # Initialize chatbot components
    tmp_dir = cl.user_session.get('tmp_dir')
    file_path = cl.user_session.get("uploaded_file_path")

    if not file_path:
        await cl.Message(content="No uploaded file found to resume the chatbot session. Please start a new session.").send()
        return

    # Check for retriever and embeddings before initializing chatbot manager
    try:
        # Initialize the retriever/embeddings manager
        embeddings_manager = EmbeddingsManager(
            model_name="BAAI/bge-small-en",
            device="cpu",
            encode_kwargs={"normalize_embeddings": True},
            qdrant_url="https://72efe8dd-cc4e-4468-9047-a6d9de74edb2.us-east4-0.gcp.cloud.qdrant.io:6333",
            collection_name="vector_db_md"
        )
        cl.user_session.set("retriever", embeddings_manager)

        # Check if embeddings are already generated for the file
        if not cl.user_session.get("embeddings_generated"):
            embeddings_manager.create_embeddings(file_path)
            cl.user_session.set("embeddings_generated", True)

        # Initialize the chatbot manager
        chatbot_manager = ChatbotManager(
            model_name="BAAI/bge-small-en",
            device="cpu",
            encode_kwargs={"normalize_embeddings": True},
            llm_model="llama3-70b-8192",
            qdrant_url="https://72efe8dd-cc4e-4468-9047-a6d9de74edb2.us-east4-0.gcp.cloud.qdrant.io:6333",
            collection_name="vector_db_md"
        )
        cl.user_session.set("chatbot_manager", chatbot_manager)
        print("Chatbot Manager Resumed Successfully")
    except Exception as e:
        await cl.Message(content=f"Error during chatbot manager initialization: {e}").send()
        return

    await cl.Message(content="Chat session resumed. You can continue asking questions!").send()

# Function to format user input into a structured prompt
def format_prompt(user_input):
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant for document-based queries.",
        },
        {
            "role": "user",
            "content": user_input
        }
    ]
 
async def invoke_llm(messages):
    chatbot_manager = cl.user_session.get('chatbot_manager', None)
    if chatbot_manager is None:
        raise ValueError("ChatbotManager is not initialized in the user session.")
    user_message = messages[-1]["content"]
    response = chatbot_manager.get_response(user_message)
    return response
 
def parse_response(response):
    if isinstance(response, dict):
        return {"message": response.get('message', 'Unable to generate a valid response.')}
    return {"message": response.strip()}
    
@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    chat_history = cl.user_session.get("chat_history")
    user_input = message.content

    messages = cl.user_session.get("messages", [])
    messages.append({"role": "user", "content": user_input})
    cl.user_session.set("messages", messages)

    chat_history = cl.user_session.get("chat_history", [])
    chat_history.append({"role": "user", "content": user_input})
    cl.user_session.set("chat_history", chat_history)

    try:
        df = cl.user_session.get("dataframe")
        if df is not None and not df.empty:
            summary = cl.user_session.get("summary")
            if not summary:
                await cl.Message(content="Please generate a summary first.").send()
                return
            graph = cl.user_session.get("graph")
            if not graph:
                graph = agentBuilder(df, summary)
                cl.user_session.set("graph", graph)
            summ, visual, code = visualizer(graph, df, summary, user_input)
            image_data = base64.b64decode(visual)
            image = cl.Image(name="visualization_image", display="inline", content=image_data)
            await cl.Message(content="Here is the visualization:", elements=[image]).send()
            await cl.Message(content=f"Summary: {summ}").send()
            save_chat_message(session_id,user_input,summ)
            tts = gTTS(str(summ), lang="en")
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_audio_file.name)
            elements = [
                cl.Audio(name="summary_audio.mp3", path=temp_audio_file.name, display="inline"),
            ]
            await cl.Message(
                content="Here is the audio summary:",
                elements=elements,
            ).send()
            temp_audio_file.close()
            os.unlink(temp_audio_file.name)
        else:
            prompt = [{"role": "user", "content": user_input}]
            response = await invoke_llm(prompt)
            answer = parse_response(response)
            messages.append({"role": "assistant", "content": answer["message"]})
            cl.user_session.set("messages", messages)
            chat_history.append({"role": "assistant", "content": answer["message"]})
            cl.user_session.set("chat_history", chat_history)
            save_chat_message(session_id,user_input,response)
            await cl.Message(content=answer["message"]).send()

            tts = gTTS(answer["message"], lang="en")
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_audio_file.name)
            elements = [
                cl.Audio(name="response.mp3", path=temp_audio_file.name, display="inline"),
            ]
            await cl.Message(
                content="Here is the audio response:",
                elements=elements,
            ).send()
            temp_audio_file.close()
            os.unlink(temp_audio_file.name)

    except Exception as e:
        error_message = f"An error occurred: {e}"
        save_chat_message(session_id, "assistant", error_message)
        await cl.Message(content=f"An error occurred: {e}").send()

## Speech query for chatbotmanager and Analytics
@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)
 
    cl.user_session.get("audio_buffer").write(chunk.data)

@cl.on_audio_end
async def on_audio_end():
    session_id = cl.user_session.get("session_id")
    audio_buffer = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)
    mime_type = cl.user_session.get("audio_mime_type")
    transcription = transcribe_audio(audio_buffer, mime_type)
    if not transcription:
        await cl.Message(content="Failed to transcribe the audio. Please try again.").send()
        return
    print(f"Transcription: {transcription}")
    await cl.Message(content=f"Transcribed Query: {transcription}").send()
    messages = cl.user_session.get("messages", [])
    messages.append({"role": "user", "content": transcription})
    cl.user_session.set("messages", messages)
    chat_history = cl.user_session.get("chat_history", [])
    chat_history.append({"role": "user", "content": transcription})
    cl.user_session.set("chat_history", chat_history)

    try:
        df = cl.user_session.get("dataframe")
        if df is not None and not df.empty:
            summary = cl.user_session.get("summary")
            if not summary:
                await cl.Message(content="Please generate a summary first.").send()
                return

            graph = cl.user_session.get("graph")
            if not graph:
                graph = agentBuilder(df, summary)
                cl.user_session.set("graph", graph)
            summ, visual,code = visualizer(graph, df, summary, transcription)
            if summ and visual:
                image_data = base64.b64decode(visual)
                image = cl.Image(name="visualization_image", display="inline", content=image_data)
                await cl.Message(content="Here is the visualization:", elements=[image]).send()
                await cl.Message(content=f"Summary: {summ}").send()
                save_chat_message(session_id,transcription,summ)
                tts = gTTS(str(summ), lang="en")
                temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tts.save(temp_audio_file.name)
                elements = [
                    cl.Audio(name="summary_audio.mp3", path=temp_audio_file.name, display="inline"),
                ]
                await cl.Message(
                    content="Here is the audio summary:",
                    elements=elements,
                ).send()
                temp_audio_file.close()
                os.unlink(temp_audio_file.name)
            else:
                await cl.Message(content="Unable to process the query for analytics.").send()
        else:
            prompt = [{"role": "user", "content": transcription}]
            response = await invoke_llm(prompt)

            if response:
                answer = parse_response(response)

                if answer and "message" in answer:
                    messages.append({"role": "assistant", "content": answer["message"]})
                    cl.user_session.set("messages", messages)
                    chat_history.append({"role": "assistant", "content": answer["message"]})
                    cl.user_session.set("chat_history", chat_history)
                    await cl.Message(content=answer["message"]).send()
                    save_chat_message(session_id,transcription,response)

                    tts = gTTS(answer["message"], lang="en")
                    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    tts.save(temp_audio_file.name)
                    elements = [
                        cl.Audio(name="response.mp3", path=temp_audio_file.name, display="inline"),
                    ]
                    await cl.Message(
                        content="Here is the audio response:",
                        elements=elements,
                    ).send()
                    temp_audio_file.close()
                    os.unlink(temp_audio_file.name)
                else:
                    await cl.Message(content="Error parsing chatbot response.").send()
            else:
                await cl.Message(content="Failed to generate a chatbot response.").send()

    except Exception as e:
        await cl.Message(content=f"An error occurred: {e}").send()


@cl.action_callback("generate_summary")
async def generate_summary_action(action: cl.Action):
    df = cl.user_session.get("dataframe")
    if df is None:
        await cl.Message(content="Please upload a file first.").send()
        return
    try:
        summary_result = summarizer(df)
        if isinstance(summary_result, dict):
            cl.user_session.set("summary", summary_result)
            await cl.Message(content="Summary generated successfully!").send()
            await cl.Message(content=json.dumps(summary_result, indent=2)).send()
            actions = [
                cl.Action(name="generate_goals", value="generate_goals", description="Generate Goals"),
            ]
            await cl.Message(content="You can now choose an action: Generate Goals", actions=actions).send()
        else:
            raise ValueError("Generated summary is not in a valid format.")
    except Exception as e:
        await cl.Message(content=f"Error generating summary: {e}").send()

@cl.action_callback("generate_goals")
async def generate_goals_action(action: cl.Action):
    summary = cl.user_session.get("summary")
    if summary is None:
        await cl.Message(content="Please generate a summary first.").send()
        return
    try:
        goals = goals_generate(summary)
        if isinstance(goals, list):
            cl.user_session.set("goals", goals)
            await cl.Message(content="Goals generated successfully!").send()
            await cl.Message(content=json.dumps(goals, indent=2)).send()
        else:
            raise ValueError("Generated goals are not in a valid format.")
    except Exception as e:
        await cl.Message(content=f"Error generating goals: {e}").send()


if __name__ == "__main__":
    initialize_chat_db()
    print("Chat history database initialized.")
