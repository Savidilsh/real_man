from typing import List, Tuple

def generate_prompt(current_location, scene_graph_str, instruction):

    prompt = f"""
    You are a robot planning assistant.  
    The robot has:  
    - A mobile base for navigation between rooms.  
    - An arm for manipulation (grasping, opening, placing).  

    You are given:  
    1. A scene graph in JSON format.  
    2. A natural language instruction describing a task for the robot.  
    3. The robot's current location.  

    Your task: Generate a high-level plan as a sequence of steps.  
    - Each step must be on its own line.  
    - Each step begins with a tag:  
        - `nav` for navigation  
        - `arm` for any arm action (open, pick, place, etc.)  
    - For `nav` steps:  
        - Output only the **target destination**.  
        - Example:  
            nav, fridge
            nav, dining_table  
    - For `arm` steps: 
        - Output a concise instruction string that should be sent to the robot's arm policy model.  
        - Example:  
            arm, Open the fridge and pick the apple inside  
            arm, Place the apple on the dining table  

    Ensure the plan follows a logical sequence that respects the topology of the scene graph.  
    Do not add explanations or extra text. Output only the plan.  

    If the instruction refers to something that cannot be done given the scene graph, or mentions an object that does not exist in the graph, do not generate a plan — simply state that this cannot be done or that the object is not found. Do not generate multiple nav commands if you are going to one destination. Just output the target destination directly. Output multiple nav commands only if going to multiple different destinations.

    ---

    ### Current location:
    {current_location}

    ### Instruction:
    {instruction}

    ### Scene graph:
    {scene_graph_str}
"""
    return prompt

def read_json_as_str(json_file_path):
    """Reads a JSON file and returns its content as a raw string."""
    with open(json_file_path, "r", encoding="utf-8") as f:
        json_as_str = f.read()
    return json_as_str



def parse_llm_response(llm_response: str) -> List[Tuple[str, str]]:
    """
    Parse LLM plan output where each line is in the format:
        nav, destination
        arm, instruction

    Returns:
        PLAN: List[Tuple[str, str]]
          Each tuple is:
            ("nav", destination)
            ("arm", instruction)
    """
    PLAN: List[Tuple[str, str]] = []

    for raw_line in llm_response.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if "," not in line:
            continue  # ignore malformed lines

        action_type, rest = line.split(",", 1)
        action_type = action_type.strip().lower()
        rest = rest.strip()

        if action_type in {"nav", "arm"} and rest:
            PLAN.append((action_type, rest))

    return PLAN