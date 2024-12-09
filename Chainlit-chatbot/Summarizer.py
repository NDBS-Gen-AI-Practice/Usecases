from dotenv import load_dotenv
_ = load_dotenv()
import os
from langchain_core.prompts import ChatPromptTemplate
import warnings
import pandas as pd
from langchain_groq import ChatGroq
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA



def summarizer(df: pd.DataFrame):
    print("inside func")
    class Summarizer():
        def __init__(self) -> None:
            self.summary = None

        def check_type(self, dtype: str, value):
            """Cast value to right type to ensure it is JSON serializable"""
            if "float" in str(dtype):
                return float(value)
            elif "int" in str(dtype):
                return int(value)
            else:
                return value

        def get_column_properties(self, df: pd.DataFrame, n_samples: int = 3) -> list[dict]:
            """Get properties of each column in a pandas DataFrame"""
            properties_list = []
            for column in df.columns:
                dtype = df[column].dtype
                properties = {}
                if dtype in [int, float, complex]:
                    properties["dtype"] = "number"
                    properties["std"] = self.check_type(dtype, df[column].std())
                    properties["min"] = self.check_type(dtype, df[column].min())
                    properties["max"] = self.check_type(dtype, df[column].max())

                elif dtype == bool:
                    properties["dtype"] = "boolean"
                elif dtype == object:
                    # Check if the string column can be cast to a valid datetime
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            pd.to_datetime(df[column], errors='raise')
                            properties["dtype"] = "date"
                    except ValueError:
                        # Check if the string column has a limited number of values
                        if df[column].nunique() / len(df[column]) < 0.5:
                            properties["dtype"] = "category"
                        else:
                            properties["dtype"] = "string"
                elif pd.api.types.is_categorical_dtype(df[column]):
                    properties["dtype"] = "category"
                elif pd.api.types.is_datetime64_any_dtype(df[column]):
                    properties["dtype"] = "date"
                else:
                    properties["dtype"] = str(dtype)

                # add min max if dtype is date
                if properties["dtype"] == "date":
                    try:
                        properties["min"] = df[column].min()
                        properties["max"] = df[column].max()
                    except TypeError:
                        cast_date_col = pd.to_datetime(df[column], errors='coerce')
                        properties["min"] = cast_date_col.min()
                        properties["max"] = cast_date_col.max()
                # Add additional properties to the output dictionary
                nunique = df[column].nunique()
                if "samples" not in properties:
                    non_null_values = df[column][df[column].notnull()].unique()
                    n_samples = min(n_samples, len(non_null_values))
                    samples = pd.Series(non_null_values).sample(
                        n_samples, random_state=42).tolist()
                    properties["samples"] = samples
                properties["num_unique_values"] = nunique
                properties["semantic_type"] = ""
                properties["description"] = ""
                properties_list.append(
                    {"column": column, "properties": properties})

            return properties_list

    # template = ChatPromptTemplate([
    # ("system", """
    # You are an experienced data analyst that can annotate datasets. Given a list of dictionary that contains the properties of columns, your instructions are as follows:
    # i) ALWAYS generate the name of the dataset and the dataset_description
    # ii) ALWAYS generate a field description.
    # You must return an updated JSON dictionary without any preamble or explanation.
    # The json should always be in backticks ``` .
    # *Properties of Columns (List of Dictionary):
    # {list}
    # """),
    # ])
    template = ChatPromptTemplate([
    ("system", """
    You are an experienced data analyst that can annotate datasets. Given a list of dictionary that contains the properties of columns, your instructions are as follows:
    i) ALWAYS generate the name of the dataset and the dataset_description
    ii) ALWAYS generate a field description.
    You must return an updated JSON dictionary without any preamble or explanation.
    The json should always be in backticks ``` .
    """),
    (
        "human",
        """
        Help summarise this
        *Properties of Columns (List of Dictionary):
        {list}
        """
    )
    ])

    summ = Summarizer()
    x = summ.get_column_properties(df)
    model = ChatGroq(api_key=os.getenv('GROQ_API_KEY'), model='gemma2-9b-it')  
    # model = ChatGroq(api_key=os.getenv('GROQ_API_KEY'), model='llama-3.1-70b-versatile')  
    # model = ChatGroq(api_key=os.getenv('GROQ_API_KEY'), model='mixtral-8x7b-32768')  
    # model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    # model = ChatNVIDIA(
    #     model="meta/llama-3.1-70b-instruct",
    #     api_key="xxxxxx", 
    #     )

    response = model.invoke(template.invoke({
        "list": x 
    }))
    print(response.content)
    pattern = r'```json\n(.*?)\n```'  # Match content between ```json and ```
    match = re.search(pattern, response.content, re.DOTALL)

    if match:
        json_str = match.group(1)  # Extract the JSON string
        try:
            json_data = json.loads(json_str)  # Parse JSON
            print("Summary ready")
            return json_data
        except json.JSONDecodeError as e:
            return(e)
    else:
        return("No JSON found in the string.")
    

