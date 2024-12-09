from dotenv import load_dotenv
_ = load_dotenv()
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
from langchain_core.prompts import ChatPromptTemplate
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import os
import re
from langchain_nvidia_ai_endpoints import ChatNVIDIA

def goals_generate(dataset_summary: dict, n=5):
        """Generate goals given a summary of data"""
        FORMAT_INSTRUCTIONS = """
        THE OUTPUT MUST BE A CODE SNIPPET OF A VALID LIST OF JSON OBJECTS. IT MUST USE THE FOLLOWING FORMAT:

        ```[
            { "index": 0,  "question": "What is the distribution of X", "visualization": "histogram of X", "rationale": "This tells about "} ..
            ]
        ```
        THE OUTPUT SHOULD ONLY USE THE JSON FORMAT ABOVE.
        """
        user_prompt = f"""The number of GOALS to generate is {n}. The goals should be based on the data summary below, \n\n .
        {dataset_summary} \n\n"""

        
        persona = "A highly skilled data analyst who can come up with complex, insightful goals about data"

        user_prompt += f"""\n The generated goals SHOULD BE FOCUSED ON THE INTERESTS AND PERSPECTIVE of a '{persona} persona, who is insterested in complex, insightful goals about the data. \n"""
        
        messages = ChatPromptTemplate([
            ("system", """
You are a an experienced data analyst who can generate a given number of insightful GOALS about data, when given a summary of the data, and a specified persona. The VISUALIZATIONS YOU RECOMMEND MUST FOLLOW VISUALIZATION BEST PRACTICES (e.g., must use bar charts instead of pie charts for comparing quantities) AND BE MEANINGFUL (e.g., plot longitude and latitude on maps where appropriate). They must also be relevant to the specified persona. Each goal must include a question, a visualization (THE VISUALIZATION MUST REFERENCE THE EXACT COLUMN FIELDS FROM THE SUMMARY), and a rationale (JUSTIFICATION FOR WHICH dataset FIELDS ARE USED and what we will learn from the visualization). Each goal MUST mention the exact fields from the dataset summary above
"""),
            (
                "human",
                """
                {user_prompt}\n\n{FORMAT_INSTRUCTIONS} \n\n.
                The generated {n} goals are: \n
                """
            )
            ])
        
        model = ChatGroq(api_key=os.getenv('GROQ_API_KEY'), model='gemma2-9b-it')
        # model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        # model = ChatGroq(api_key=os.getenv('GROQ_API_KEY'), model='llama-3.2-11b-vision-preview')
        # model = ChatNVIDIA(
        # model="meta/llama-3.1-70b-instruct",
        # api_key="xxxxxx", 
        # )

        response = model.invoke(messages.invoke({ "FORMAT_INSTRUCTIONS": FORMAT_INSTRUCTIONS,
                                                 "n": '5',
                                                 "user_prompt": user_prompt}))

        json_string = response.content
        print(json_string)
        pattern = r'```json\n(.*?)\n```'  # Match content between ```json and ```
        match = re.search(pattern, response.content, re.DOTALL)

        if match:
            json_str = match.group(1)  # Extract the JSON string
            try:
                json_data = json.loads(json_str)  # Parse JSON
                print("Goals Ready")
                return json_data
            except json.JSONDecodeError as e:
                return(e)
        else:
            return("No JSON found in the string.")