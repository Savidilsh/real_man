from __future__ import annotations

import argparse
import json
import re
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_TOOL_RE = re.compile(r"^\s*(\w+)\s*\(\s*(.*?)\s*\)\s*$")


@dataclass
class Segment:
    path: List[str]
    tool_sequence: List[str]
    destination: str


@dataclass
class ParseResult:
    segments: List[Segment]
    grasp_objects: List[str]
    has_non_nav_tools: bool
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _extract_arg(arg_text: str) -> str:
    if not arg_text:
        return ""
    match = re.search(r'"([^"]+)"', arg_text)
    if match:
        return match.group(1).strip()
    match = re.search(r"'([^']+)'", arg_text)
    if match:
        return match.group(1).strip()
    arg_text = arg_text.strip()
    if "," in arg_text:
        arg_text = arg_text.split(",", 1)[0].strip()
    return arg_text


def parse_gpt_output(raw_text: str) -> ParseResult:
    """
    Parse GPT tool output and extract nav segments + grasp objects.
    Rules:
      - Ignore speak(), vqa(), face(), greet(), place()
      - pass() is not allowed in CV mode
      - Support multiple nav() segments
      - Each segment must end with nav()
    """
    segments: List[Segment] = []
    current_path: List[str] = []
    current_tools: List[str] = []
    grasp_objects: List[str] = []
    has_non_nav_tools = False

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _TOOL_RE.match(line)
        if not match:
            continue

        tool, arg_text = match.group(1).lower(), match.group(2)
        if tool in {"speak", "vqa", "face", "greet", "place"}:
            has_non_nav_tools = True
            continue

        if tool == "show":
            return ParseResult([], [], has_non_nav_tools, "show() is not allowed in CV mode.")

        if tool == "pass":
            return ParseResult([], [], has_non_nav_tools, "pass() is not allowed in CV mode; use nav() only.")

        if tool == "nav":
            arg = _extract_arg(arg_text)
            if not arg:
                return ParseResult([], [], has_non_nav_tools, f"Missing argument for {tool}()")
            current_path.append(arg)
            current_tools.append(tool)
            segments.append(Segment(current_path, current_tools, current_path[-1]))
            current_path = []
            current_tools = []
            continue

        if tool == "grasp":
            arg = _extract_arg(arg_text)
            if not arg:
                return ParseResult([], [], has_non_nav_tools, "Missing argument for grasp()")
            grasp_objects.append(arg)
            has_non_nav_tools = True

    if current_path:
        return ParseResult([], [], has_non_nav_tools, "Segment missing nav() at the end")
    if not segments:
        if has_non_nav_tools:
            return ParseResult([], grasp_objects, True, None)
        return ParseResult([], [], has_non_nav_tools, "Missing nav()")

    return ParseResult(segments, grasp_objects, has_non_nav_tools, None)


def build_name_map(scene_nodes: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    return {name.lower(): name for name in scene_nodes.keys()}


def load_scene_graph(scene_path: str | Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(Path(scene_path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "nodes" in data:
        nodes = data["nodes"]
    elif isinstance(data, list):
        nodes = data
    else:
        raise ValueError("Scene graph JSON must be a list or a dict with a 'nodes' key.")

    scene_nodes: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        name = str(node.get("node_name") or node.get("name") or "").strip()
        if not name:
            continue

        neighbors = node.get("neighbor_nodes")
        if neighbors is None:
            neighbors = node.get("neighbor_room")
        neighbors_list = [str(n) for n in _ensure_list(neighbors) if n is not None]

        marker = str(node.get("node_marker") or node.get("marker") or "S").upper()
        if marker not in {"S", "O"}:
            marker = "S"

        child_nodes = node.get("child_nodes") or []

        scene_nodes[name] = {
            "neighbors": neighbors_list,
            "child_nodes": child_nodes,
            "marker": marker,
        }

    name_map = build_name_map(scene_nodes)
    for name, info in scene_nodes.items():
        normalized = []
        for nb in info["neighbors"]:
            normalized.append(name_map.get(nb.lower(), nb))
        info["neighbors"] = normalized

    return scene_nodes


def _resolve_name(name: str, name_map: Dict[str, str]) -> Optional[str]:
    if name is None:
        return None
    key = str(name).strip().lower()
    return name_map.get(key)


def _resolve_path(path: List[str], name_map: Dict[str, str]) -> Tuple[Optional[List[str]], Optional[str]]:
    resolved: List[str] = []
    for node in path:
        resolved_name = _resolve_name(node, name_map)
        if resolved_name is None:
            return None, node
        resolved.append(resolved_name)
    return resolved, None


def build_graph(scene_nodes: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    graph: Dict[str, List[str]] = {}
    for name, info in scene_nodes.items():
        graph[name] = list(info.get("neighbors") or [])
    return graph


def bfs_shortest(
    start: str,
    goal: str,
    graph: Dict[str, List[str]],
    markers: Optional[Dict[str, str]] = None,
    junction_only: bool = False,
) -> Optional[List[str]]:
    if start not in graph or goal not in graph:
        return None
    if start == goal:
        return []

    q = deque([[start]])
    visited = {start}

    def _allowed(node: str) -> bool:
        if not junction_only or markers is None:
            return True
        if node == start or node == goal:
            return True
        return markers.get(node, "S") == "O"

    while q:
        path = q.popleft()
        node = path[-1]
        if node == goal:
            return path[1:]

        for n in graph.get(node, []):
            if n not in visited and _allowed(n):
                visited.add(n)
                q.append(path + [n])

    return None


def verify_nodes_exist(path: List[str], scene_nodes: Dict[str, Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    for n in path:
        if n not in scene_nodes:
            return False, f"Node does not exist: {n}"
    return True, None


def verify_adjacency(
    start: str,
    path: List[str],
    scene_nodes: Dict[str, Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    current = start
    for nxt in path:
        neighbors = scene_nodes.get(current, {}).get("neighbors") or []
        if nxt not in neighbors:
            return False, f"Illegal transition {current} -> {nxt}"
        current = nxt
    return True, None


def _collect_objects_from_children(children: List[Dict[str, Any]], objects: set[str], all_nodes: set[str]) -> None:
    for child in children:
        name = child.get("node_name")
        if name:
            all_nodes.add(name)
            ctype = str(child.get("node_type", "")).lower()
            if ctype == "object":
                objects.add(name)
        grand = child.get("child_nodes") or []
        if grand:
            _collect_objects_from_children(grand, objects, all_nodes)


def collect_scene_objects(scene_nodes: Dict[str, Dict[str, Any]]) -> Tuple[set[str], set[str]]:
    objects: set[str] = set()
    all_nodes: set[str] = set(scene_nodes.keys())
    for node in scene_nodes.values():
        children = node.get("child_nodes") or []
        _collect_objects_from_children(children, objects, all_nodes)
    if not objects:
        objects = set(all_nodes)
    return objects, all_nodes


def build_retry_prompt(reason: str) -> str:
    return (
        "The plan is invalid.\n"
        f"Reason: {reason}.\n"
        "Fix the plan.\n"
        "Output format:\n"
        'REASON: <one short sentence>\n'
        "Then tool calls only, one per line."
    )


def run_simulator(
    gpt_output: str,
    current_location: str,
    scene_nodes: Dict[str, Dict[str, Any]],
    required_object: Optional[str] = None,
) -> Dict[str, Any]:
    """End-to-end validation for a GPT plan using only the scene graph."""
    parse_result = parse_gpt_output(gpt_output)
    if not parse_result.ok:
        return {"status": "INVALID", "reason": parse_result.error}

    name_map = build_name_map(scene_nodes)
    start = _resolve_name(current_location, name_map)
    if start is None:
        return {"status": "INVALID", "reason": f"Start node does not exist: {current_location}"}

    graph = build_graph(scene_nodes)
    markers = {name: info.get("marker", "S") for name, info in scene_nodes.items()}
    current = start
    flat_path: List[str] = []
    flat_tools: List[str] = []
    resolved_segments: List[Dict[str, Any]] = []

    if not parse_result.segments:
        objects, _ = collect_scene_objects(scene_nodes)
        objects_map = {name.lower(): name for name in objects}
        for obj in parse_result.grasp_objects:
            key = obj.strip().lower()
            if key not in objects_map:
                return {"status": "INVALID", "reason": f"Object not present in scene graph: {obj}"}
        if required_object:
            key = required_object.strip().lower()
            if key not in objects_map:
                return {"status": "INVALID", "reason": f"Object not present in scene graph: {required_object}"}
        if parse_result.has_non_nav_tools:
            return {
                "status": "VALID",
                "path": [],
                "destination": current,
                "tool_sequence": [],
                "segments": [],
                "grasp_objects": parse_result.grasp_objects,
            }
        return {"status": "INVALID", "reason": "Missing nav()"}

    for seg_idx, segment in enumerate(parse_result.segments, start=1):
        path, bad_node = _resolve_path(segment.path, name_map)
        if path is None or not path:
            return {"status": "INVALID", "reason": f"Node does not exist: {bad_node}"}

        ok, reason = verify_nodes_exist(path, scene_nodes)
        if not ok:
            return {"status": "INVALID", "reason": reason}

        destination = path[-1]
        # CV mode treats each nav("destination") as a high-level goto command.
        # Validate reachability rather than requiring explicit intermediate nodes.
        path = [destination]
        expected_path = bfs_shortest(current, destination, graph, markers=markers, junction_only=True)
        expected_path_type = "o_only"
        if expected_path is None:
            expected_path = bfs_shortest(current, destination, graph, markers=markers, junction_only=False)
            expected_path_type = "fallback" if expected_path is not None else "none"
        if expected_path is None:
            return {
                "status": "INVALID",
                "reason": f"No path found for segment {seg_idx}",
                "segment": seg_idx,
                "segment_start": current,
                "segment_destination": destination,
                "expected_path": expected_path,
                "expected_path_type": expected_path_type,
            }
        # CV validator is intentionally lighter than the default simulator.

        resolved_segments.append(
            {
                "path": path,
                "tool_sequence": segment.tool_sequence,
                "destination": destination,
                "expected_path": expected_path,
                "expected_path_type": expected_path_type,
            }
        )
        flat_path.extend(path)
        flat_tools.extend(segment.tool_sequence)
        current = destination

    # In CV mode, grasp() is placeholder speech behavior, so we do not enforce
    # strict object-name membership against scene_graph child_nodes.

    return {
        "status": "VALID",
        "path": flat_path,
        "destination": current,
        "tool_sequence": flat_tools,
        "segments": resolved_segments,
        "grasp_objects": parse_result.grasp_objects,
    }


def _read_gpt_text(args: argparse.Namespace) -> str:
    if args.gpt_file:
        return Path(args.gpt_file).read_text(encoding="utf-8")
    if args.gpt:
        return args.gpt
    return sys.stdin.read()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate GPT navigation output (single-file).")
    parser.add_argument("--scene", required=True, help="Path to scene graph JSON")
    parser.add_argument("--start", required=True, help="Current location node_name")
    parser.add_argument("--end", help="Destination node_name to compute shortest path")
    parser.add_argument("--gpt", help="Raw GPT output text (optional)")
    parser.add_argument("--gpt-file", help="Path to file containing GPT output")
    parser.add_argument("--required-object", help="Object required in the scene graph")
    args = parser.parse_args()

    scene_nodes = load_scene_graph(args.scene)
    name_map = build_name_map(scene_nodes)
    start = _resolve_name(args.start, name_map)
    if start is None:
        print(json.dumps({"status": "INVALID", "reason": f"Start node does not exist: {args.start}"}))
        return 2

    if args.end:
        destination = _resolve_name(args.end, name_map)
        if destination is None:
            print(json.dumps({"status": "INVALID", "reason": f"End node does not exist: {args.end}"}))
            return 2
        graph = build_graph(scene_nodes)
        markers = {name: info.get("marker", "S") for name, info in scene_nodes.items()}
        path = bfs_shortest(start, destination, graph, markers=markers, junction_only=True)
        fallback = None
        status = "OK"
        if path is None:
            status = "NO_O_ONLY_PATH"
            fallback = bfs_shortest(start, destination, graph, markers=markers, junction_only=False)
        print(json.dumps(
            {
                "status": status,
                "start": start,
                "destination": destination,
                "path": path,
                "fallback_path": fallback,
            },
            indent=2,
        ))
        return 0

    gpt_text = _read_gpt_text(args)
    if not gpt_text.strip():
        print("No GPT output provided.", file=sys.stderr)
        return 2

    result = run_simulator(
        gpt_output=gpt_text,
        current_location=args.start,
        scene_nodes=scene_nodes,
        required_object=args.required_object,
    )

    print(json.dumps(result, indent=2))
    if result.get("status") == "INVALID":
        print("\n--- Retry Prompt ---")
        print(build_retry_prompt(result.get("reason", "Unknown error")))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
