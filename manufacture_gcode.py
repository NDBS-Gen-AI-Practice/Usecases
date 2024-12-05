from crewai import Agent, Task, Crew, Process
from interpreter import interpreter
from langchain.tools import tool
from langchain_openai import ChatOpenAI
import os

os.environ['OPENAI_API_KEY']='openai_api_key'

# 1. Configuration and Tools
llm = ChatOpenAI(model="gpt-4-turbo-preview")
interpreter.auto_run = True
interpreter.llm.model = "openai/gpt-4-turbo-preview"

object=input("Enter the object : ")

class CLITool:
    @tool("Executor")
    def execute_cli_command(command: str):
        """Create and Execute code using Open Interpreter."""
        result = interpreter.chat(command)
        return result

# 2. Creating an Agent for CLI tasks
cli_agent = Agent(
    role='Software Engineer',
    goal='Always use Executor Tool. Ability to perform CLI operations, write programs and execute using Exector Tool',
    backstory="""
    Expert in command line operations, creating and executing code.
    You are an experienced trimesh Python developer like
    when the object to create is golf ball you will produce a file like the below
    import trimesh
    import numpy as np

    # Create the golf ball base (a sphere)
    radius = 2  # Radius of the golf ball
    golf_ball = trimesh.creation.icosphere(subdivisions=4, radius=radius)

    # Parameters for the dimples
    dimple_radius = 0.15  # Radius of each dimple
    num_dimples = 200  # Approximate number of dimples

    # Create dimples on the golf ball
    dimples = []
    np.random.seed(42)  # For reproducibility

    for _ in range(num_dimples):
        # Randomly position dimples on the surface of the golf ball
        direction = trimesh.unitize(np.random.normal(size=3))  # Random direction vector on sphere
        dimple_center = direction * (radius - dimple_radius)  # Place dimple slightly inside the surface

        # Create the dimple as a small sphere
        dimple = trimesh.creation.icosphere(subdivisions=2, radius=dimple_radius)
        dimple.apply_translation(dimple_center)

        # Add the dimple to the list
        dimples.append(dimple)

    # Combine all the dimples and the base sphere
    golf_ball_with_dimples = golf_ball

    for dimple in dimples:
        # Subtract the dimple from the golf ball to create indentations
        golf_ball_with_dimples = golf_ball_with_dimples.difference(dimple)

    # Export the golf ball as an STL file
    golf_ball_with_dimples.export('object.stl')


    """,
    tools=[CLITool.execute_cli_command],
    verbose=True,
    llm=llm 
)


# 3. Defining a Task for CLI operations
cli_task = Task(
    description=f'Every necessary packagess and libraries have been installed, excuted the code for {object} output should be in object.stl',
    agent=cli_agent,
    tools=[CLITool.execute_cli_command],
    expected_output="""
    Confirmation that the object.stl file generated.  )

    """
)


# 4. Creating a Crew with CLI focus
cli_crew = Crew(
    agents=[cli_agent],
    tasks=[cli_task],
    process=Process.sequential,
    manager_llm=llm
)

# 5. Run the Crew
result = cli_crew.kickoff()
print(result)

import subprocess
slic3r_path = r".venv\slicer\Slic3r-1.3.0.64bit\Slic3r.exe"
subprocess.run([slic3r_path, "object.stl", "--output", "cube.gcode"])