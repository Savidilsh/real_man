#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plan-only script: Generate task plans using LLM without ROS execution.
This script targets agent_cv behavior and tool contract for planning tests.
"""

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
STATE_FILE = BASE_DIR / "plan_only_cv_test_state.json"
DEFAULT_SCENE_GRAPH_PATH = BASE_DIR / "cv_new.json"
DEFAULT_START_LOCATION = "meeting_room_1"

from simulator_cv import run_simulator, build_retry_prompt, load_scene_graph



def generate_agent_prompt(current_location, held_object, scene_graph_str, instruction):
    prompt = f"""You are a robot planner. Convert the instruction into tool calls only.

Output rules:
- First line must be: REASON: <one short sentence>.
- After that, output only tool calls, one per line.
- Allowed: nav("..."), speak("..."), vqa("..."), face("..."), greet(), grasp("...")
- Use exact scene graph node_name in nav().
- For object/tool selection, use scene-graph child_nodes whenever possible.
- Keep speak() short and natural.

Navigation rules:
- Use nav() only (no pass()).
- Use direct destination nav() calls (e.g., nav("kitchen")), not junction-by-junction nav chains.

VQA/check workflow rules:
- For inspection/status tasks that use vqa(), remember the initial `current_location`.
- Run checks as nav("...") + vqa("...") steps.
- Do NOT add speak() lines that claim visual findings right after vqa().
- After all checks, return to the initial location with nav("<initial current_location>").
- Final speak() after returning must be neutral (e.g., "I finished the checks and will report what I observed.").

Cleanup workflow rules:
- For spill/cleanup/mess instructions, use a cleanup sequence rather than hand-over by default.
- Pick a suitable cleaning item from child_nodes via grasp("..."), then perform imagined cleaning at the target location with speak().
- After cleaning, if `garbage_bin` exists in the scene graph, nav("garbage_bin") and speak() that you are imagining disposing the used item.
- Only hand over items to a person if the user explicitly asks for bring/give/hand-over.

Grasp placeholder rules:
- grasp() is symbolic only; no real arm movement.
- Use exact child-node object names when available (e.g., "cup"), not vague targets like "water".
- If the requested thing is a liquid/serving (not directly graspable), choose a suitable container/tool child node first, then use speak() to describe the imagined preparation/transfer.
- After grasp("object"), add speak() saying pickup is imagined.
- For bring/give/hand-over tasks, after reaching recipient, add:
  speak("I am handing over my imagined <object> to you.")

Scenario example:
User: "Prof Ian needs something to drink."
Plan example:
speak("I will get a drink for Prof Ian.")
nav("water_dispenser_1")
grasp("cup")
speak("If I had arms, I would pick up this cup and imagine filling it with water.")
nav("prof_ian_room")
speak("I am handing over my imagined cup of water to you.")

CURRENT STATE:
- current_location: "{current_location}"
- held_object: "{held_object if held_object else 'None'}"

SCENE GRAPH JSON:
{scene_graph_str}

USER INSTRUCTION:
"{instruction}"

Plan:
"""
    return prompt


def read_json_as_str(path: str) -> str:
    """Read JSON file and return as formatted string."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error loading scene graph: {e}"


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_anchor_response_id(client: OpenAI, scene_graph_path: str, model: str) -> str | None:
    scene_path = str(scene_graph_path)
    state = _load_state()
    file_hash = _file_sha256(scene_path)
    cached = state.get("scene_graph", {})
    cached_id = cached.get("anchor_response_id")
    if cached.get("path") == scene_path and cached.get("sha256") == file_hash and cached_id:
        return cached_id

    try:
        scene_graph_str = read_json_as_str(scene_path)
        seed_prompt = (
            "Scene graph JSON (memorize for future planning):\n"
            f"{scene_graph_str}\n"
            "Reply only with OK."
        )
        response = client.responses.create(
            model=model,
            input=[{"role": "user", "content": [{"type": "input_text", "text": seed_prompt}]}],
        )
        anchor_id = getattr(response, "id", None)
        if anchor_id:
            state["scene_graph"] = {
                "path": scene_path,
                "sha256": file_hash,
                "anchor_response_id": anchor_id,
                "updated_at": int(time.time()),
            }
            _save_state(state)
        return anchor_id
    except Exception:
        return None


def _extract_response_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    try:
        return response.output[0].content[0].text
    except Exception:
        return ""


def _call_llm(
    client: OpenAI,
    prompt_text: str,
    model: str,
    temperature: float,
    previous_response_id: str | None,
) -> tuple[str | None, str | None, bool]:
    if hasattr(client, "responses"):
        response = client.responses.create(
            model=model,
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt_text}]}],
            temperature=temperature,
            previous_response_id=previous_response_id,
        )
        return _extract_response_text(response), getattr(response, "id", None), True

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=temperature,
    )
    return response.choices[0].message.content, None, False


def parse_plan(plan_text: str) -> list:
    """Parse LLM-returned plan in 'action(args)' format"""
    parsed_plan = []
    for line in plan_text.strip().splitlines():
        match = re.match(r"(\w+)\((.*)\)", line.strip())
        if match:
            action, args_str = match.groups()
            # Simple parameter processing, remove quotes
            args = args_str.strip().replace("'", "").replace('"', '')
            parsed_plan.append({"action": action, "args": args if args else None})
    return parsed_plan


def _format_path(path):
    if path is None:
        return None
    if not isinstance(path, list):
        return str(path)
    if not path:
        return "[]"
    return " -> ".join(path)


def _format_validation_failure(result: dict) -> str:
    if not result:
        return "Unknown validation error."

    reason = result.get("reason") or "Unknown validation error."
    seg_start = result.get("segment_start")
    seg_dest = result.get("segment_destination")
    expected_path = result.get("expected_path")
    given_path = result.get("given_path")

    expected_str = _format_path(expected_path)
    given_str = _format_path(given_path)

    if seg_start and seg_dest and seg_start == seg_dest:
        return f"Already at {seg_dest}."
    if expected_str == "[]" and seg_dest:
        return f"Already at {seg_dest}."

    if reason.startswith("Illegal transition"):
        base = reason
        details = []
        if seg_start and seg_dest:
            details.append(f"From {seg_start} to {seg_dest}.")
        if expected_str is not None:
            details.append(f"Expected path: {expected_str}.")
        return " ".join([base] + details)

    if reason.startswith("Path is not shortest") or reason.startswith("Path differs from expected shortest"):
        details = []
        if seg_start and seg_dest:
            details.append(f"From {seg_start} to {seg_dest}.")
        if expected_str is not None:
            details.append(f"Expected path: {expected_str}.")
        if given_str is not None:
            details.append(f"Given path: {given_str}.")
        return " ".join(details) if details else reason

    if reason.startswith("Pass node must be O:"):
        node = reason.split(":", 1)[-1].strip()
        return f"Invalid pass node {node}. Use pass() only for O nodes unless no O-only path exists."

    if reason.startswith("Node does not exist:"):
        node = reason.split(":", 1)[-1].strip()
        return f"Unknown node {node}. Use exact node_name from the scene graph."

    if reason.startswith("Start node does not exist:"):
        node = reason.split(":", 1)[-1].strip()
        return f"Unknown start node {node}. Check current_location."

    if reason in ("Missing nav()", "Segment missing nav() at the end"):
        return "Missing nav(): every segment must end with nav()."

    if reason.startswith("Missing argument for"):
        return reason

    if reason.startswith("pass() is not allowed in CV mode"):
        return "pass() is not allowed in CV mode. Use nav() only."

    if reason.startswith("show() is not allowed in CV mode"):
        return "show() is not allowed in CV mode."

    if reason.startswith("Object not present in scene graph:"):
        obj = reason.split(":", 1)[-1].strip()
        return f"Object not present: {obj}."

    if reason.startswith("No path found for segment"):
        if seg_start and seg_dest:
            return f"No path from {seg_start} to {seg_dest}."
        return reason

    parts = [reason]
    if seg_start and seg_dest:
        parts.append(f"From {seg_start} to {seg_dest}.")
    if expected_str is not None:
        parts.append(f"Expected path: {expected_str}.")
    if given_str is not None:
        parts.append(f"Given path: {given_str}.")
    return " ".join(parts)


def main():
    """Interactive planning loop."""
    print("\n=== Robot Plan Generator (No Execution, agent_cv Prompt) ===")
    print("This script generates plans without executing them.")
    print("Type 'exit' or 'quit' to stop.\n")
    
    # Default settings
    scene_graph_path = str(DEFAULT_SCENE_GRAPH_PATH)
    current_location = DEFAULT_START_LOCATION
    held_object = None
    openai_model = "gpt-5.2"
    temperature = 0.2
    
    print(f"Scene Graph: {scene_graph_path}")
    print(f"Starting Location: {current_location}")
    print(f"OpenAI Model: {openai_model}\n")

    scene_nodes = None
    try:
        scene_nodes = load_scene_graph(scene_graph_path)
    except Exception as e:
        print(f"[Validation] Disabled: {e}")

    client = OpenAI()
    supports_responses = hasattr(client, "responses")
    anchor_response_id = None
    if supports_responses:
        anchor_response_id = _get_anchor_response_id(client, scene_graph_path, openai_model)
    
    while True:
        try:
            instruction = input("> Instruction: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if instruction.lower() in {"exit", "quit", "q"}:
            break
        
        if not instruction:
            continue
        
        print("\nGenerating plan...")
        if anchor_response_id:
            scene_graph_str = "(scene graph provided earlier)"
        else:
            scene_graph_str = read_json_as_str(scene_graph_path)
        base_prompt = generate_agent_prompt(current_location, held_object, scene_graph_str, instruction)
        max_attempts = 3
        retry_reason = None
        previous_response_id = anchor_response_id
        use_conversation_state = supports_responses

        for attempt in range(1, max_attempts + 1):
            prompt = base_prompt
            if retry_reason:
                if use_conversation_state:
                    prompt = build_retry_prompt(retry_reason)
                else:
                    prompt = base_prompt + "\n\n" + build_retry_prompt(retry_reason)

            try:
                plan_text, response_id, used_state = _call_llm(
                    client=client,
                    prompt_text=prompt,
                    model=openai_model,
                    temperature=temperature,
                    previous_response_id=previous_response_id,
                )
            except Exception as e:
                print(f"Error calling OpenAI API: {e}")
                break

            use_conversation_state = used_state
            if response_id:
                previous_response_id = response_id

            if plan_text is None:
                break
            print("\n[Generated Plan]")
            print(plan_text)

            if scene_nodes is None:
                break

            validation_result = run_simulator(
                gpt_output=plan_text,
                current_location=current_location,
                scene_nodes=scene_nodes,
            )
            if validation_result.get("status") == "VALID":
                break

            retry_reason = _format_validation_failure(validation_result)
            print("\n[Plan Invalid]")
            print(retry_reason)
            if attempt == max_attempts:
                print("\n[Failed after retries]")
            else:
                print("\nRetrying...")

        print()


if __name__ == '__main__':
    main()
