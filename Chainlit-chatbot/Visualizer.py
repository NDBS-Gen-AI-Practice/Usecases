from dotenv import load_dotenv
_ = load_dotenv()
import os
from AgentBuilder import agentBuilder
import pandas as pd
import matplotlib.pyplot as plt

def visualizer(graph, df: pd.DataFrame, dataset_summary, task):
    thread = {"configurable": {"thread_id": "1"}}
    for s in graph.stream({
        'task': task,
    }, thread):
        s
    
    current_values = graph.get_state(thread)

    # all_states = []
    # for state in graph.get_state_history(thread):
    #     print(state.values['task'])
    #     # print(state.next)
    #     all_states.append(state)
    #     print("--")

    return current_values.values['summary'], current_values.values['visual'], current_values.values['code']