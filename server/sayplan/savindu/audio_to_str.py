#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio-to-Text Transcription Server
Receives audio from phone via WebSocket, transcribes using Whisper,
and coordinates plan verification with robot agent.
"""
 
 
import asyncio
import websockets
import whisper
import os
import tempfile
import threading
import rospy
from std_msgs.msg import String as RosString
import json
from pathlib import Path
from datetime import datetime
 
# Configuration - save JSON files in json/ subfolder
BASE_DIR = Path(__file__).resolve().parent
JSON_DIR = BASE_DIR / "json"
JSON_DIR.mkdir(exist_ok=True)  # Create json/ folder if it doesn't exist
 
AUDIO_TEXT_FILE = JSON_DIR / "latest_audio_text.json"
AUDIO_STATUS_FILE = JSON_DIR / "audio_status.json"
PLAN_REQUEST_FILE = JSON_DIR / "plan_request.json"
PLAN_RESPONSE_FILE = JSON_DIR / "plan_response.json"
MIC_CONTROL_FILE = JSON_DIR / "mic_control.json"
 
# Global set to track all connected clients
connected_clients = set()
pending_transcription = {}
ros_lock = threading.Lock()
ros_instruction = "Waiting for instruction..."
ros_plan_lines = []
ros_current_step = -1
ros_stop_pub = None
 
# We'll set this once asyncio loop is running
async_loop = None
agent_state_queue = None
 
 
def compute_step_status(plan, current_idx: int):
    """Return list of per-step statuses exactly like viz.py coloring logic."""
    out = []
    for i, step in enumerate(plan):
        if i < current_idx:
            status = "executed"   # green
        elif i == current_idx:
            status = "current"    # yellow
        else:
            status = "pending"    # white
        out.append({"index": i, "step": step, "status": status})
    return out
 
 
async def broadcast_agent_state():
    """Send current agent state to all connected phones."""
    with ros_lock:
        instr = ros_instruction
        plan = list(ros_plan_lines)
        idx = int(ros_current_step)
 
    msg = json.dumps({
        "type": "agent_state",
        "instruction": instr,
        "plan": plan,
        "current_step": idx,
        "step_status": compute_step_status(plan, idx),
        "timestamp": datetime.now().isoformat()
    })
 
    disconnected = set()
    for client in connected_clients:
        try:
            await client.send(msg)
        except websockets.exceptions.ConnectionClosed:
            disconnected.add(client)
 
    connected_clients.difference_update(disconnected)
    return len(connected_clients) > 0
 
 
async def agent_state_sender_loop():
    """Consumes ROS updates and broadcasts them to phone."""
    # Drain updates quickly and only send latest burst
    while True:
        try:
            # wait for at least one update signal
            await agent_state_queue.get()
 
            # drain any extra pending signals (coalesce)
            while not agent_state_queue.empty():
                agent_state_queue.get_nowait()
 
            # broadcast latest snapshot
            await broadcast_agent_state()
 
        except Exception as e:
            print(f"[ERROR] agent_state_sender_loop: {e}")
 
        await asyncio.sleep(0.01)
 
def _notify_agent_state_changed():
    """Thread-safe: wake asyncio sender loop from ROS callback threads."""
    global async_loop, agent_state_queue
    if async_loop is None or agent_state_queue is None:
        return
    try:
        async_loop.call_soon_threadsafe(agent_state_queue.put_nowait, 1)
    except Exception:
        pass
 
 
def ros_instruction_cb(msg: RosString):
    global ros_instruction, ros_plan_lines, ros_current_step
    with ros_lock:
        ros_instruction = msg.data
        ros_plan_lines = []
        ros_current_step = -1
    _notify_agent_state_changed()
 
 
def ros_plan_cb(msg: RosString):
    global ros_plan_lines
    with ros_lock:
        if msg.data:
            ros_plan_lines = msg.data.splitlines()
        else:
            ros_plan_lines = []
    _notify_agent_state_changed()
 
 
def ros_current_step_cb(msg: RosString):
    global ros_current_step
    try:
        idx = int(msg.data)
    except Exception:
        return
    with ros_lock:
        ros_current_step = idx
    _notify_agent_state_changed()
 
 
def init_ros_subscribers():
    """
    Start ROS node + subscribe to agent topics.
    IMPORTANT: disable_signals=True so ROS doesn't fight asyncio signal handling.
    """
    if not rospy.core.is_initialized():
        rospy.init_node("audio_to_str_server", anonymous=True, disable_signals=True)
 
    global ros_stop_pub
    rospy.Subscriber("/agent/instruction", RosString, ros_instruction_cb, queue_size=10)
    rospy.Subscriber("/agent/plan", RosString, ros_plan_cb, queue_size=10)
    rospy.Subscriber("/agent/current_step", RosString, ros_current_step_cb, queue_size=10)
    ros_stop_pub = rospy.Publisher("/agent/stop", RosString, queue_size=1)

    print("[ROS] Subscribed to /agent/instruction, /agent/plan, /agent/current_step")
    print("[ROS] Ready to publish /agent/stop")


def publish_stop_request():
    if ros_stop_pub is None:
        print("[ROS] Stop publisher not initialized.")
        return False
    try:
        ros_stop_pub.publish(RosString(data="stop"))
        print("[ROS] Stop request published to /agent/stop")
        return True
    except Exception as e:
        print(f"[ROS] Failed to publish stop request: {e}")
        return False
 
def initialize_audio_server():
    """Initialize the audio server - create status files"""
    status_data = {
        "status": "initializing",
        "timestamp": datetime.now().isoformat(),
        "server_ip": "100.104.233.12",
        "server_port": 8765
    }
    with open(AUDIO_STATUS_FILE, 'w') as f:
        json.dump(status_data, f, indent=2)
   
    # Clear any previous text
    text_data = {
        "text": "",
        "timestamp": datetime.now().isoformat(),
        "available": False
    }
    with open(AUDIO_TEXT_FILE, 'w') as f:
        json.dump(text_data, f, indent=2)
   
    # Clear plan files
    plan_request = {
        "plan": [],
        "instruction": "",
        "timestamp": datetime.now().isoformat(),
        "waiting_response": False
    }
    with open(PLAN_REQUEST_FILE, 'w') as f:
        json.dump(plan_request, f, indent=2)
   
    plan_response = {
        "response": "",
        "timestamp": datetime.now().isoformat(),
        "available": False
    }
    with open(PLAN_RESPONSE_FILE, 'w') as f:
        json.dump(plan_response, f, indent=2)
 
    mic_control = {
        "blocked": False,
        "timestamp": datetime.now().isoformat(),
        "available": False
    }
    with open(MIC_CONTROL_FILE, 'w') as f:
        json.dump(mic_control, f, indent=2)
   
    print(f"[*] Audio server initialized")
    print(f"[*] Status file: {AUDIO_STATUS_FILE}")
    print(f"[*] Text output: {AUDIO_TEXT_FILE}")
    print(f"[*] Plan request: {PLAN_REQUEST_FILE}")
    print(f"[*] Plan response: {PLAN_RESPONSE_FILE}")
 
def update_audio_status(status):
    """Update the server status"""
    status_data = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "server_ip": "100.104.233.12",
        "server_port": 8765
    }
    with open(AUDIO_STATUS_FILE, 'w') as f:
        json.dump(status_data, f, indent=2)
 
def save_transcribed_text(text):
    """Save the transcribed text to a file for other scripts to read"""
    text_data = {
        "text": text,
        "timestamp": datetime.now().isoformat(),
        "available": True
    }
    with open(AUDIO_TEXT_FILE, 'w') as f:
        json.dump(text_data, f, indent=2)
 
def save_plan_response(response):
    """Save the plan verification response from phone"""
    response_data = {
        "response": response,
        "timestamp": datetime.now().isoformat(),
        "available": True
    }
    with open(PLAN_RESPONSE_FILE, 'w') as f:
        json.dump(response_data, f, indent=2)
 
print("[*] Loading Whisper model...")
model = whisper.load_model("base.en")  
print("[✓] Model loaded. Ready for phone connections.")
 
async def handle_audio(websocket):
    """Handle WebSocket connection from phone"""
    print(f"[PHONE] Connected: {websocket.remote_address}")
    connected_clients.add(websocket)
    update_audio_status("connected")
   
    try:
        async for message in websocket:
            # Handle binary audio data
            if isinstance(message, bytes):
                print(f"[PHONE] Received audio: {len(message)} bytes")
                update_audio_status("connected")
 
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                    temp_wav.write(message)
                    temp_path = temp_wav.name
 
                try:
                    print(f"[WHISPER] Transcribing audio...")
                    result = model.transcribe(
                        temp_path,
                        language="en",
                        task="transcribe",
                        fp16=False
                    )
                    text = result["text"].strip()
                    print(f"[WHISPER] Transcribed: '{text}'")
                   
                    pending_transcription[websocket] = {
                        "text": text,
                        "timestamp": datetime.now().isoformat()
                    }
 
                    # Send transcription back to phone
                    await websocket.send(json.dumps({
                        "type": "transcription",
                        "text": text
                    }))
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
           
            # Handle text messages (plan responses)
            elif isinstance(message, str):
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                   
                    if msg_type == "plan_response":
                        response = data.get("response", "").strip().lower()
                        print(f"[PHONE] Plan response: {response}")
                        save_plan_response(response)
                       
                        # Acknowledge receipt
                        await websocket.send(json.dumps({
                            "type": "plan_response_ack",
                            "received": response
                        }))
                    if msg_type == "transcription_verification":
                        resp = data.get("response", "").strip().lower()
                        print(f"[PHONE] Transcription verification: {resp}")
 
                        pending = pending_transcription.get(websocket)
 
                        if resp == "confirm" and pending:
                            save_transcribed_text(pending["text"])
                            print(f"[WHISPER] Saved confirmed transcription: '{pending['text']}'")
 
                            await websocket.send(json.dumps({
                                "type": "transcription_verification_ack",
                                "status": "saved"
                            }))
 
                        elif resp == "reject":
                            pending_transcription.pop(websocket, None)
 
                            await websocket.send(json.dumps({
                                "type": "transcription_verification_ack",
                                "status": "discarded"
                            }))
 
                        else:
                            await websocket.send(json.dumps({
                                "type": "transcription_verification_ack",
                                "status": "no_pending"
                            }))
                    if msg_type == "stop":
                        print("[PHONE] Stop requested by app")
                        ok = publish_stop_request()
                        await websocket.send(json.dumps({
                            "type": "stop_ack",
                            "ok": bool(ok)
                        }))
                   
                except json.JSONDecodeError:
                    print(f"[PHONE] Received non-JSON text: {message}")
                    # Treat as simple yes/no response
                    response = message.strip().lower()
                    if response in ["yes", "no", "y", "n"]:
                        print(f"[PHONE] Simple response: {response}")
                        save_plan_response(response)
 
    except websockets.exceptions.ConnectionClosed:
        print(f"[PHONE] Disconnected: {websocket.remote_address}")
        pending_transcription.pop(websocket, None)
        connected_clients.discard(websocket)
        if not connected_clients:
            update_audio_status("disconnected")
 
async def send_plan_to_clients(plan_data):
    """Send plan to all connected clients (phone)"""
    if not connected_clients:
        print("[!] No clients connected to send plan")
        return False
   
    message = json.dumps({
        "type": "plan_verification",
        "instruction": plan_data["instruction"],
        "plan": plan_data["plan"],
        "timestamp": datetime.now().isoformat()
    })
   
    disconnected = set()
    for client in connected_clients:
        try:
            await client.send(message)
            print(f"[PHONE] Sent plan to: {client.remote_address}")
        except websockets.exceptions.ConnectionClosed:
            disconnected.add(client)
   
    # Remove disconnected clients
    connected_clients.difference_update(disconnected)
    return len(connected_clients) > 0
 
 
async def send_mic_control_to_clients(control_data):
    """Send mic control to all connected clients (phone)."""
    if not connected_clients:
        return False
    message = json.dumps({
        "type": "mic_control",
        "blocked": bool(control_data.get("blocked", False)),
        "timestamp": datetime.now().isoformat()
    })
    disconnected = set()
    for client in connected_clients:
        try:
            await client.send(message)
        except websockets.exceptions.ConnectionClosed:
            disconnected.add(client)
    connected_clients.difference_update(disconnected)
    return len(connected_clients) > 0
 
async def plan_sender_loop():
    """Background task that monitors plan requests and sends them to phone"""
    while True:
        try:
            if PLAN_REQUEST_FILE.exists():
                with open(PLAN_REQUEST_FILE, 'r') as f:
                    request = json.load(f)
               
                if request.get("waiting_response", False):
                    # Send plan to connected clients
                    await send_plan_to_clients(request)
                   
                    # Mark as sent
                    request["waiting_response"] = False
                    request["sent"] = True
                    with open(PLAN_REQUEST_FILE, 'w') as f:
                        json.dump(request, f, indent=2)
            if MIC_CONTROL_FILE.exists():
                with open(MIC_CONTROL_FILE, 'r') as f:
                    mic_control = json.load(f)
                if mic_control.get("available", False):
                    await send_mic_control_to_clients(mic_control)
                    mic_control["available"] = False
                    with open(MIC_CONTROL_FILE, 'w') as f:
                        json.dump(mic_control, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Plan sender loop: {e}")
       
        await asyncio.sleep(0.5)
 
async def main():
    """Main server loop"""
    global async_loop, agent_state_queue
 
    initialize_audio_server()
    update_audio_status("running")
 
    # Capture asyncio loop for thread-safe notifications
    async_loop = asyncio.get_running_loop()
    agent_state_queue = asyncio.Queue()
 
    # Start ROS subscribers (no JSON needed, same truth as viz.py)
    init_ros_subscribers()
 
    # Start background tasks
    plan_sender_task = asyncio.create_task(plan_sender_loop())
    agent_state_task = asyncio.create_task(agent_state_sender_loop())
 
    async with websockets.serve(handle_audio, "100.104.233.12", 8765):
        print("\n" + "="*60)
        print("WHISPER AUDIO-TO-TEXT SERVER RUNNING")
        print("="*60)
        print(f"WebSocket: ws://100.104.233.12:8765")
        print(f"Status: Ready to receive phone audio")
        print("="*60 + "\n")
        await asyncio.Future()
 
if __name__ == "__main__":
    asyncio.run(main())
 
 

