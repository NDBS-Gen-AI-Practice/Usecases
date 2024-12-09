from dotenv import load_dotenv
_ = load_dotenv()
import os
import pandas as pd
import matplotlib.pyplot as plt
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Optional
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langgraph.checkpoint.memory import MemorySaver
import re
from langchain_groq import ChatGroq
import os
import io
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA


def agentBuilder(df: pd.DataFrame, dataset_summary):
    PLAN_PROMPT = """ You are a helpful assistant highly skilled in writing PERFECT code for visualizations. \
    Given some code template, you complete the template to generate a visualization given the dataset and the goal described. \
    The visualization code MUST only use data fields that exist in the dataset summary or fields that are \
    transformations based on existing field_names. \
    Only use variables that have been defined in the code or are in the dataset summary. \
    You MUST first generate a brief plan for how you would solve the task \
    e.g. what transformations you would apply e.g. if you need to construct a new column, what fields you would use, \
    what visualization type you would use, what aesthetics you would use, etc. \
    The dataset is always stores in df do not assume anything else and do not load any other dataset. \
    Below is the dataset summary
    ------
    {dataset_summary}
    """

    CODER_PROMPT = """ You are a helpful assistant highly skilled in writing PERFECT code for visualizations in PYTHON using matplotlib. \
    The code you write MUST FOLLOW VISUALIZATION BEST PRACTICES ie. meet the specified goal, \
    apply the right transformation, \
    use the right visualization type, use the right data encoding, and use the right aesthetics (e.g., ensure axis are legible).\
    The transformations you apply MUST be correct and the fields you use MUST be correct. \
    The dataset is always stored in df. \
    The visualization CODE MUST BE CORRECT and MUST NOT CONTAIN ANY SYNTAX OR LOGIC ERRORS \
    (e.g., it must consider the field types and use them correctly). \
    Also remember that the code you generate will be given in the exec() function of python. \
    Always add a legend with various colors where appropriate.  \
    Always add a line in the end to save the visual to a variable named 'visual' using gcf, eg: 'visual = plt.gcf()'\
    You MUST return a FULL PYTHON PROGRAM that starts with an import statement ENCLOSED IN BACKTICKS ```  \
    The dataset is always stores in df do not assume anything else and do not load any other dataset. \
    DO NOT add any explanation."""

    SUMM_PROMPT = """ Summarize the visualization you just created and explain its purpose. \
    Describe how the data was transformed and why, the visualization type used and why it is suitable for the data, \
    the encoding of data (e.g., axis, colors, shapes), and how the visualization helps meet the goal specified. \
    Provide clear insights or inferences that can be drawn from the visual, explaining what the trends, patterns, or relationships suggest. \
    Ensure your explanation is concise, clear, and highlights how the visualization aligns with best practices. \
    Use technical terms appropriately but keep the explanation accessible for a general audience."""

    memory = MemorySaver()

    class AgentState(TypedDict):
        task: str
        plan: str
        code: str
        visual: str
        summary: str

    model = ChatGroq(api_key=os.getenv('GROQ_API_KEY'), model='gemma2-9b-it')
    # model = ChatGroq(api_key=os.getenv('GROQ_API_KEY'), model='llama-3.1-70b-versatile')
    # model = ChatGroq(api_key=os.getenv('GROQ_API_KEY'), model='mixtral-8x7b-32768')
    # model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    # model = ChatNVIDIA(
    #     model="meta/llama-3.1-70b-instruct",
    #     api_key="xxxxx", 
    #     )


    def plan_node(state: AgentState):
        messages = [
            SystemMessage(content=PLAN_PROMPT.format(dataset_summary=dataset_summary)), 
            HumanMessage(content=state['task'])
        ]
        response = model.invoke(messages)
        return {"plan": response.content}
    
    def extract_python_code(model, messages):
        while True:
            # Invoke the model
            response = model.invoke(messages)
        
            # Define the regex pattern for Python code in backticks
            # pattern = r'```(.*?)```'
            pattern = r'```python?\s*(.*?)```'
            match = re.search(pattern, response.content, re.DOTALL)

            if match:
                python_code = match.group(1).strip()
                print("Extracted Python Code:")
                print(python_code)
                return python_code  # Exit loop and return the code
            else:
                print("No Python code found. Retrying...")


    def coder_node(state: AgentState):
        user_message = HumanMessage(
            content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}\n\nHere is dataset Summary:\n\n{dataset_summary}")
        messages = [
            SystemMessage(
                content=CODER_PROMPT
            ),
            user_message
            ]
        response = extract_python_code(model, messages)
        # print(response)
        return {
            "code": response
        }
    
    def execution_node(state: AgentState):
        # Create a namespace to execute the code in a controlled environment
        exec_namespace = {}
        exec(state['code'], {"df": df}, exec_namespace)
    
        if 'visual' in exec_namespace:
            plot_buffer = io.BytesIO() 
            figure = exec_namespace['visual'] 
            if hasattr(figure, "savefig"):
                figure.savefig(plot_buffer, format='png') 
                plt.close(figure)  
                plot_buffer.seek(0)
                visual_base64 = base64.b64encode(plot_buffer.getvalue()).decode('utf-8')
                return {
                    'visual': visual_base64
                }
            else:
                raise TypeError("'visual' is not a valid Figure object.")
        else:
            raise RuntimeError("The code did not save a visual in a variable named 'visual'.")


        
    def summary_node(state: AgentState):
        user_message = HumanMessage(
            content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}\n\nHere is dataset summary:\n\n{dataset_summary}")
        messages = [
            SystemMessage(
                content=SUMM_PROMPT
            ),
            user_message
            ]
        response = model.invoke(messages)
        # print(response)
        return {
            "summary": response.content
        }
    
    builder = StateGraph(AgentState)
    builder.add_node("planner", plan_node)
    builder.add_node("coder", coder_node)
    builder.add_node("executer", execution_node)
    builder.add_node("summ", summary_node)
    builder.set_entry_point("planner")
    builder.add_edge("planner", "coder")
    builder.add_edge("coder", "executer")
    builder.add_edge("executer", "summ")
    graph = builder.compile(checkpointer=memory)
    print("Agent Ready")
    return graph

    

    




