#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice-controlled Robot Agent
Uses phone audio input for instructions and plan verification.
No terminal fallback - pure voice/phone operation.
"""


import rospy
import time
import math
import json
import threading
import cv2
import os
import re
import numpy as np
import base64
import sys
import select
from collections import deque


from std_msgs.msg import String, Float64
from sensor_msgs.msg import CompressedImage
from servo_ros.msg import ServoMove
from agv_ros.msg import NavigationJoyControl
from pathlib import Path
from openai import OpenAI
from sayplan.utils import read_json_as_str
from head_base_rotation import (
    HeadBaseRotator,
    SERVO2_MIN,
    SERVO2_MAX,
    SERVO2_CENTER,
    UNITS_PER_DEG,
    SETTLE_TIME,
    STOP_DEG,
    HEAD_HOLD_DELAY,
    BASE_START_DELAY
)


import torch
from datetime import datetime
import tempfile
import uuid

try:
    import soundfile as sf
except ImportError:
    sf = None


# Audio integration paths: read/write from json/ subfolder (same as audio_to_str.py)
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
from simulator_cv import run_simulator, build_retry_prompt, load_scene_graph
from plan_only_cv import (
    _format_validation_failure,
    _get_anchor_response_id,
)
# --- Conversation History and Topic Classification ---
def classify_topic_change(new_instruction: str, conversation_history: list) -> str:
    """Classify if new instruction is a continuation or a new topic."""
    if not conversation_history:
        return "NEW_TOPIC"

    history_summary = ""
    for turn in conversation_history[-6:]:
        role = turn.get("role", "")
        content = turn.get("content", "")[:100]
        history_summary += f"{role}: {content}\n"

    prompt = f"""You are a conversation analyzer. Determine if the new user input is:
1. CONTINUATION - Related to or following up on the previous conversation
2. NEW_TOPIC - Completely different subject that requires fresh context, or if you went to exit a person.

OUTPUT FORMAT (respond with EXACTLY ONE LINE):
CLASSIFICATION: <CONTINUATION|NEW_TOPIC>

Examples:

Previous: "I'm hungry" -> "Go to kitchen"
New: "What's in the fridge?" -> CLASSIFICATION: CONTINUATION

Previous: "Go to kitchen" -> "Here's the kitchen"
New: "Tell me about robotics" -> CLASSIFICATION: NEW_TOPIC

Previous: "Show me robot arms" -> "Which one?"
New: "The one with grippers" -> CLASSIFICATION: CONTINUATION

Previous: "What's in the space chamber?" -> "Space equipment"
New: "I need to find a restroom" -> CLASSIFICATION: NEW_TOPIC

Previous: "Take me to the lab" -> "Going to lab"
New: "What can I do here?" -> CLASSIFICATION: CONTINUATION

RECENT CONVERSATION:
{history_summary}

NEW USER INPUT:
"{new_instruction}"

CLASSIFICATION:"""
    return prompt
JSON_DIR = BASE_DIR / "json"
JSON_DIR.mkdir(exist_ok=True)  # Create json/ folder if it doesn't exist

AUDIO_TEXT_FILE = JSON_DIR / "latest_audio_text.json"
AUDIO_STATUS_FILE = JSON_DIR / "audio_status.json"
STATUS_FILE = JSON_DIR / "status.json"
LOCATION_FILE = JSON_DIR / "current_location.json"
INTERRUPT_ANSWER_FILE = JSON_DIR / "interrupt_answer.json"
INTERRUPT_ANSWERS_FILE = JSON_DIR / "interrupt_answers.json"
PLAN_REQUEST_FILE = JSON_DIR / "plan_request.json"
PLAN_RESPONSE_FILE = JSON_DIR / "plan_response.json"
CV_SCENE_GRAPH_PATH = BASE_DIR / "cv_new.json"
CV_START_LOCATION = "meeting_room_1"

TTS_MODEL_NAME = "tts_models/en/ljspeech/vits"
TTS_FALLBACK_MODEL_NAME = "tts_models/en/ljspeech/glow-tts"
TTS_SPEAKER = None
NAV_FAILURE_ASSIST_DELAY_SECONDS_DEFAULT = 40.0
NAV_ASSIST_WAIT_SECONDS = 5.0
PREDEFINED_AUDIO_INTERVAL_SEC = 10.0
NAV_PROGRESS_TIMEOUT_SEC = 40.0
SERVO1_MIN = 450
SERVO1_MAX = 570
SERVO1_CENTER = 510
POSE_POLL_HZ =1.0
STOP_BURST_COUNT = 4
STOP_BURST_DELAY = 0.05
POSE_STALE_MAX_SEC = 8.0
BASE_ANG_VEL = 0.6
# SERVO2_*, UNITS_PER_DEG, SETTLE_TIME, STOP_DEG imported from head_base_rotation
STOP_DEG = 2.0
_PLAN_LINE_RE = re.compile(r'^(\w+)\((.*?)\)\s*$')


def is_audio_available():
  """Check if audio system is running and has recent text available."""
  try:
      if not AUDIO_STATUS_FILE.exists():
          return False

      with open(AUDIO_STATUS_FILE, 'r') as f:
          status = json.load(f)

      if status.get('status') not in ['running', 'connected']:
          return False

      timestamp_str = status.get('timestamp', '')
      if timestamp_str:
          timestamp = datetime.fromisoformat(timestamp_str)
          time_diff = (datetime.now() - timestamp).total_seconds()
          if time_diff > 300:
              return False

      return True
  except Exception as e:
      print(f"Error checking audio availability: {e}")
      return False


def get_audio_text():
  """Get the latest transcribed text from audio system."""
  try:
      if not AUDIO_TEXT_FILE.exists():
          return None

      with open(AUDIO_TEXT_FILE, 'r') as f:
          data = json.load(f)

      if not data.get('available', False):
          return None

      text = data.get('text', '').strip()

      data['available'] = False
      with open(AUDIO_TEXT_FILE, 'w') as f:
          json.dump(data, f, indent=2)

      return text if text else None
  except Exception as e:
      print(f"Error reading audio text: {e}")
      return None


def _read_terminal_instruction():
  if not sys.stdin or not sys.stdin.isatty():
      return None
  try:
      readable, _, _ = select.select([sys.stdin], [], [], 0)
  except (OSError, ValueError):
      return None
  if not readable:
      return None
  text = sys.stdin.readline()
  if text == "":
      return None
  text = text.strip()
  return text or None


def is_exit_instruction(instruction):
  return bool(instruction and instruction.strip().lower() == "exit")


def cleanup_transient_json_state():
  now = datetime.now().isoformat()
  payloads = {
      AUDIO_TEXT_FILE: {
          "text": "",
          "timestamp": now,
          "available": False,
      },
      INTERRUPT_ANSWER_FILE: {
          "answer": "",
          "question_id": "",
          "timestamp": now,
          "available": False,
      },
      INTERRUPT_ANSWERS_FILE: {
          "answer": "",
          "question_id": "",
          "timestamp": now,
          "available": False,
      },
      PLAN_REQUEST_FILE: {
          "request": "",
          "timestamp": now,
          "available": False,
      },
      PLAN_RESPONSE_FILE: {
          "response": "",
          "timestamp": now,
          "available": False,
      },
      STATUS_FILE: {
          "status": "idle",
          "timestamp": now,
          "available": False,
      },
  }
  for path, payload in payloads.items():
      try:
          path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
      except Exception as exc:
          print(f"Warning: failed to clean {path}: {exc}")


def get_audio_instruction(prompt="> Instruction: "):
  """
  Block until an instruction is received from the phone audio system or terminal.
  """
  print(f"{prompt}[Listening for voice input or terminal input...]", flush=True)
  while True:
      try:
          terminal_text = _read_terminal_instruction()
          if terminal_text:
              print(f"[Terminal Input] {terminal_text}")
              return terminal_text

          if is_audio_available():
              audio_text = get_audio_text()
              if audio_text:
                  print(f"[Voice Input] {audio_text}")
                  return audio_text
          time.sleep(0.5)
      except KeyboardInterrupt:
          raise
      except Exception:
          time.sleep(0.5)



def _load_persisted_location(default_location=None):
  try:
      if not LOCATION_FILE.exists():
          return default_location
      data = json.loads(LOCATION_FILE.read_text(encoding="utf-8"))
      location = data.get("current_location") or data.get("location")
      return location or default_location
  except Exception:
      return default_location


def _save_persisted_location(location):
  if not location:
      return
  try:
      payload = {
          "current_location": location,
          "timestamp": datetime.now().isoformat()
      }
      LOCATION_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
  except Exception:
      pass



def generate_agent_prompt(current_location, scene_graph_str, instruction, conversation_history=None, held_object=None):
    prompt = f"""You are a robot planner. Convert the instruction into tool calls only.

Output rules:
- First line must be: REASON: <one short sentence>.
- After that, output only tool calls, one per line.
- Allowed: nav("..."), speak("..."), vqa("..."), face("..."), greet(), grasp("..."), handing("..."), throwing("...")
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

Cleanup workflow rules:
- For spill/cleanup/mess instructions, use a cleanup sequence rather than hand-over by default.
- Pick a suitable cleaning item from child_nodes via grasp("..."), then perform imagined cleaning at the target location with speak().
- After cleaning, if a garbage bin node exists in the scene graph, nav() to that exact node_name and use throwing("object") for the used item.
- Only hand over items to a person if the user explicitly asks for bring/give/hand-over.

Symbolic action rules:
- grasp(), handing(), and throwing() are symbolic only; no real arm movement.
- Use exact child-node object names when available (e.g., "cup"), not vague targets like "water".
- If the requested thing is a liquid/serving (not directly graspable), choose a suitable container/tool child node first, then use speak() to describe the imagined preparation/transfer.
- Do not add speak() immediately after grasp(), handing(), or throwing(); these tools already speak their imagined action.
- For bring/give/hand-over tasks, after reaching the recipient, use handing("object").

Scenario example:
User: "Prof Ian needs something to drink."
Plan example:
speak("I will get a drink for Prof Ian.")
nav("water_dispenser_1")
grasp("cup")
nav("prof_ian_room")
handing("cup")

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


class RobotAgent:
  """
  An integrated robot agent class for unified management and invocation of core robot functions,
  including image and pose acquisition, navigation task execution, and servo control.
  """
  def __init__(self):
      if not rospy.core.is_initialized():
          rospy.init_node('robot_agent', anonymous=True)
      rospy.loginfo("RobotAgent is initializing...")


      self.lock = threading.Lock()
      self._speak_lock = threading.Lock()
      self.image_received_event = threading.Event()


      self.current_pose = None
      self.navigation_status = 'idle'
      self.cruise_markers = []
      self.cruise_index = -1
      self.cruise_active = False
      self.cruise_done = False
      self.current_nav_target = None
      self.last_reached_marker = None
      self.latest_rgb_image = None
      self._pose_stamp = None
      self._face_servo1 = SERVO1_CENTER
      self._face_servo2 = SERVO2_CENTER
      self._imagined_grasp_object = None


      # ROS publishers
      self.pub_get_status = rospy.Publisher('/navigation_get_robot_status', String, queue_size=1)
      self.pub_nav_marker = rospy.Publisher('/navigation_marker', String, queue_size=1)
      self.pub_nav_multipoint = rospy.Publisher('/navigation_multipoint', String, queue_size=1)
      self.pub_nav_cancel = rospy.Publisher('/navigation_move_cancel', String, queue_size=1)
      self.pub_nav_joy = rospy.Publisher('/navigation_joy_control', NavigationJoyControl, queue_size=10)
      self.sub_feedback = rospy.Subscriber('/navigation_feedback', String, self._feedback_callback)
      self._pose_poll_thread = threading.Thread(target=self._pose_poll_loop, daemon=True)
      self._pose_poll_thread.start()


      self.pub_servo_control = rospy.Publisher("/servo_control/move", ServoMove, queue_size=10)


      color_topic = "/camera/color/image_raw/compressed"
      self.image_subscriber = rospy.Subscriber(
          color_topic, CompressedImage, self._image_callback, queue_size=1
      )


      self.pub_max_linear_speed = rospy.Publisher('/navigation_max_speed_linear', Float64, queue_size=1)
      self.pub_max_angular_speed = rospy.Publisher('/navigation_max_speed_angular', Float64, queue_size=1)
      self.pub_robot_audio = rospy.Publisher('/robot_audio', String, queue_size=1)
      self.sub_robot_audio_done = rospy.Subscriber('/robot_audio_done', String, self._audio_done_callback)


      self._audio_done_lock = threading.Lock()
      self._audio_done_events = {}
      self._predefined_audio_lock = threading.Lock()
      self._last_predefined_audio_time = 0.0
      self._busy_audio_stop = threading.Event()
      self._busy_audio_thread = None
      self._nav_assist_played = False

      # Initialize Coqui TTS
      rospy.loginfo("Initializing Coqui TTS...")
      try:
          from TTS.api import TTS
          import os

          # Use local models directory
          models_dir = BASE_DIR / "models"
          models_dir.mkdir(exist_ok=True)
          os.environ['TTS_HOME'] = str(models_dir)

          # Detect device for TTS
          tts_device = "cuda" if torch.cuda.is_available() else "cpu"
          rospy.loginfo(f"[TTS] Using device: {tts_device}")
          rospy.loginfo(f"[TTS] CUDA available: {torch.cuda.is_available()}")
          if torch.cuda.is_available():
              rospy.loginfo(f"[TTS] GPU: {torch.cuda.get_device_name(0)}")
          rospy.loginfo(f"[TTS] Loading Coqui TTS models from: {models_dir}")
          rospy.loginfo("[TTS] (this may take a moment on first run)...")

          self.tts_pipeline = None
          self.tts_model_name = None
          model_candidates = [TTS_MODEL_NAME]
          if TTS_FALLBACK_MODEL_NAME and TTS_FALLBACK_MODEL_NAME not in model_candidates:
              model_candidates.append(TTS_FALLBACK_MODEL_NAME)

          for model_name in model_candidates:
              try:
                  rospy.loginfo(f"[TTS] Loading model: {model_name}")
                  self.tts_pipeline = TTS(
                      model_name=model_name,
                      progress_bar=False,
                      gpu=(tts_device=="cuda")
                  )
                  self.tts_model_name = model_name
                  break
              except Exception as e:
                  rospy.logerr(f"[TTS] Failed to load model {model_name}: {e}")
                  self.tts_pipeline = None

          if not self.tts_pipeline:
              raise RuntimeError("All TTS model loads failed.")

          self.tts_speaker = TTS_SPEAKER
          rospy.loginfo(f"[TTS] Coqui TTS ready with model: {self.tts_model_name}")
      except ImportError as e:
          rospy.logerr(f"[TTS] ERROR: TTS not installed: {e}")
          rospy.logerr("[TTS] Run: pip install TTS")
          self.tts_pipeline = None
      except Exception as e:
          rospy.logerr(f"[TTS] ERROR: Failed to initialize Coqui TTS: {e}")
          rospy.logerr(f"[TTS] Error details: {type(e).__name__}: {str(e)}")
          import traceback
          rospy.logerr(f"[TTS] Traceback: {traceback.format_exc()}")
          self.tts_pipeline = None

      self.openai_client = OpenAI()
      self.vqa_model = "gpt-4o"
      rospy.loginfo("OpenAI client for VQA initialized successfully.")

      # Initialize base rotator (uses its own navigation_feedback subscription)
      self._rotator = HeadBaseRotator()
      rospy.loginfo("HeadBaseRotator initialized successfully.")

      rospy.loginfo("Initializing Arm and MoveIt control...")


      rospy.loginfo("Waiting for ROS interfaces to establish connection...")
      rospy.sleep(2)
      rospy.loginfo("RobotAgent initialization complete, ready to receive commands.")

  def _feedback_callback(self, msg):
      if not msg or not msg.data:
          return
      try:
          json_objects = msg.data.strip().split("\n")
          for json_str in json_objects:
              if not json_str:
                  continue
              # Quick check before parsing - skip if not robot_status or notification
              if '"/api/robot_status"' not in json_str and '"/api/navigation_status"' not in json_str and '"notification"' not in json_str:
                  continue
              
              data = json.loads(json_str)
              command = data.get('command') or data.get('commanWd')
              
              # Fast path for pose updates - most common case
              if command == '/api/robot_status' and 'results' in data:
                  pose_data = data['results'].get('current_pose')
                  if pose_data:
                      with self.lock:
                          self.current_pose = {
                              'x': pose_data.get('x'),
                              'y': pose_data.get('y'),
                              'theta': pose_data.get('theta')
                          }
                          self._pose_stamp = time.time()
                  continue  # Skip rest of processing for pose updates
              
              # Slower path for navigation events - less common
              with self.lock:
                  if command == '/api/navigation_status' and 'results' in data:
                      status = data['results'].get('navigation_status')
                      if status and self.navigation_status != status:
                          rospy.loginfo(
                              f"Navigation status updated via '/api/navigation_status': {self.navigation_status} -> {status}"
                          )
                          self.navigation_status = status
                  elif data.get('type') == 'notification' and 'code' in data:
                      code = data.get('code')
                      detail = data.get('data') or {}
                      description = data.get('description') or ""

                      if code == '01101':
                          markers = detail.get('markers') or []
                          if markers:
                              self.cruise_markers = list(markers)
                              self.cruise_index = -1
                              self.cruise_active = True
                              self.cruise_done = False
                              rospy.loginfo(f"[NAV] Cruise started: {','.join(markers)}")
                          self.navigation_status = 'running'
                      elif code == '01102':
                          if self.cruise_active:
                              self.cruise_active = False
                          self.cruise_done = True
                          rospy.loginfo("[NAV] Cruise finished.")
                          self.navigation_status = 'succeeded'
                      elif code == '01104':
                          if self.cruise_active:
                              self.cruise_active = False
                          self.cruise_done = False
                          reason = description or "Cruise canceled."
                          rospy.logwarn(f"[NAV] Cruise canceled: {reason}")
                          self.navigation_status = 'failed'
                      elif code == '01001':
                          target = detail.get('target')
                          if target:
                              self.current_nav_target = target
                              if self.cruise_active and self.cruise_markers and target in self.cruise_markers:
                                  idx = self.cruise_markers.index(target)
                                  self.cruise_index = idx
                                  rospy.loginfo(
                                      f"[NAV] Moving: {target} (step {idx + 1}/{len(self.cruise_markers)})"
                                  )
                              else:
                                  rospy.loginfo(f"[NAV] Moving: {target}")
                          self.navigation_status = 'running'
                      elif code == '01002':
                          target = detail.get('target') or self.current_nav_target
                          if target:
                              self.last_reached_marker = target
                              if self.cruise_markers and target in self.cruise_markers:
                                  idx = self.cruise_markers.index(target)
                                  self.cruise_index = idx
                                  rospy.loginfo(
                                      f"[NAV] Reached: {target} (step {idx + 1}/{len(self.cruise_markers)})"
                                  )
                              else:
                                  rospy.loginfo(f"[NAV] Reached: {target}")
                          if not self.cruise_active:
                              self.navigation_status = 'succeeded'
                      elif code in ('01005', '01006'):
                          reason = description or "Move retried."
                          rospy.logwarn(f"[NAV] Move retry: {reason}")
                          self.navigation_status = 'running'
                      elif code == '01004':
                          reason = description or "Move canceled."
                          rospy.logwarn(f"[NAV] Move canceled: {reason}")
                          self.navigation_status = 'failed'
                      elif code == '02003':
                          rospy.logwarn("[NAV] Estop on.")
                          self.cruise_active = False
                          self.cruise_done = False
                          self.navigation_status = 'failed'
                      elif code == '02004':
                          rospy.loginfo("[NAV] Estop off.")
      except (json.JSONDecodeError, KeyError):
          pass

  def _pose_poll_loop(self):
      rate = rospy.Rate(POSE_POLL_HZ)
      while not rospy.is_shutdown():
          self.pub_get_status.publish(String(data=""))
          rate.sleep()

  def _audio_done_callback(self, msg):
      if not msg or not msg.data:
          return
      audio_id = msg.data.strip()
      if not audio_id:
          return
      with self._audio_done_lock:
          evt = self._audio_done_events.pop(audio_id, None)
      if evt:
          evt.set()


  def _image_callback(self, color_msg):
      try:
          np_arr_color = np.frombuffer(color_msg.data, np.uint8)
          rgb_image = cv2.imdecode(np_arr_color, cv2.IMREAD_COLOR)
          with self.lock:
              if rgb_image is not None:
                  self.latest_rgb_image = rgb_image

              if rgb_image is not None and not self.image_received_event.is_set():
                  rospy.loginfo("Successfully received and decoded first RGB image frame.")
                  self.image_received_event.set()
              else:
                  if rgb_image is None:
                      rospy.logwarn_throttle(1, "Failed to decode compressed color image.")
      except Exception as e:
          rospy.logerr(f"Error processing compressed image data: {e}")

  # def get_nearest_marker_from_map(self, map_path=None):
  #     """Return the nearest marker name from the provided map JSON using the latest pose.
  #     Returns None if no pose or markers are available.
  #     """
  #     try:
  #         with self.lock:
  #             pose = self.current_pose
  #         if not pose:
  #             return None
  #
  #         if map_path is None:
  #             map_path = str(BASE_DIR.parent / "map_new_markers.json")
  #         with open(map_path, 'r', encoding='utf-8') as f:
  #             mdata = json.load(f)
  #         markers = mdata.get('markers') or []
  #         if not markers:
  #             return None
  #
  #         x = pose.get('x')
  #         y = pose.get('y')
  #         if x is None or y is None:
  #             return None
  #
  #         best = None
  #         best_d = float('inf')
  #         for m in markers:
  #             try:
  #                 mx = float(m.get('x', 0))
  #                 my = float(m.get('y', 0))
  #             except Exception:
  #                 continue
  #             d = math.hypot(x - mx, y - my)
  #             if d < best_d:
  #                 best_d = d
  #                 best = m
  #
  #         if best:
  #             return best.get('name')
  #     except Exception as e:
  #         rospy.logwarn(f"get_nearest_marker_from_map failed: {e}")
  #     return None

  def _wait_for_navigation_completion(self, timeout, interrupt_handler=None):
      start_time = rospy.get_time()
      rate = rospy.Rate(2)
      last_progress_time = start_time
      last_reached_marker = None


      with self.lock:
          self.navigation_status = 'idle'
          self._nav_assist_played = False

      wait_for_running_start = rospy.get_time()
      while rospy.get_time() - wait_for_running_start < 5:
          with self.lock:
              if self.navigation_status == 'running':
                  rospy.loginfo("Navigation task has started executing...")
                  last_progress_time = rospy.get_time()
                  last_reached_marker = self.last_reached_marker
                  break
          rate.sleep()

      while rospy.get_time() - start_time < timeout:
          if interrupt_handler is not None:
              interrupt_handler()
          with self.lock:
              status = self.navigation_status
              reached = self.last_reached_marker
              cruise_active = self.cruise_active

          if status == 'running' and cruise_active:
              if reached and reached != last_reached_marker:
                  last_reached_marker = reached
                  last_progress_time = rospy.get_time()
              if (rospy.get_time() - last_progress_time) > NAV_PROGRESS_TIMEOUT_SEC:
                  rospy.logwarn("Navigation progress timeout. Requesting assist.")
                  self.cancel_nav()
                  self._nav_assist_played = True
                  self.play_predefined_audio('assist')
                  return False

          if status == 'succeeded':
              rospy.loginfo("Navigation task succeeded!")
              return True

          if status == 'failed' or status == 'aborted':
              rospy.logwarn(f"Navigation task failed with status: '{status}'. Sending cancel command.")
              self.cancel_nav()
              return False

          rate.sleep()

      rospy.logwarn(f"Navigation task timed out after {timeout} seconds. Sending cancel command.")
      self.cancel_nav()
      return False


  def cancel_nav(self):
      rospy.loginfo("Sending navigation cancel command...")
      self.pub_nav_cancel.publish(String(data="cancel"))
      return True


  def set_max_linear_speed(self, speed):
      if not 0.1 <= speed <= 1.0:
          rospy.logwarn(f"Desired linear speed {speed} m/s is outside the recommended range [0.1, 1.0].")
      rospy.loginfo(f"Setting max linear speed to {speed} m/s...")
      self.pub_max_linear_speed.publish(Float64(data=float(speed)))
      return True


  def set_max_angular_speed(self, speed):
      if not 0.5 <= speed <= 3.5:
          rospy.logwarn(f"Desired angular speed {speed} rad/s is outside the recommended range [0.5, 3.5].")
      rospy.loginfo(f"Setting max angular speed to {speed} rad/s...")
      self.pub_max_angular_speed.publish(Float64(data=float(speed)))
      return True


  def get_rgbd(self, timeout=2.0):
      if not self.image_received_event.wait(timeout):
          rospy.logwarn("Timeout getting RGB image.")
          return None, None

      with self.lock:
          if self.latest_rgb_image is None:
              return None, None

          rgb_image = self.latest_rgb_image.copy()
          depth_image = None


      ret, jpeg_buffer = cv2.imencode('.jpg', rgb_image)
      if not ret:
          rospy.logwarn("JPEG encoding failed.")
          return None, None

      return jpeg_buffer.tobytes(), depth_image


  def nav_multipoint(self, markers: list[str], timeout: float = 90.0, interrupt_handler=None) -> bool:
      if not markers:
          rospy.logwarn("Multipoint navigation requested with empty marker list.")
          return False
      with self.lock:
          self.cruise_markers = list(markers)
          self.cruise_index = -1
          self.cruise_active = True
          self.cruise_done = False
          self.current_nav_target = None
          self.last_reached_marker = None
      marker_str = ",".join(markers)
      rospy.loginfo(f"Starting multipoint navigation: {marker_str}")
      self.pub_nav_multipoint.publish(String(data=marker_str))
      return self._wait_for_navigation_completion(timeout, interrupt_handler)


  def nav(self, location: str, timeout: float = 500.0, interrupt_handler=None) -> bool:
      rospy.loginfo(f"Atomic operation [nav]: Navigating to '{location}'")
      with self.lock:
          self.cruise_markers = [location]
          self.cruise_index = -1
          self.cruise_active = False
          self.cruise_done = False
          self.current_nav_target = location
          self.last_reached_marker = None
      return self.point_nav(location, timeout, interrupt_handler)


  def point_nav(self, point_name, timeout=400.0, interrupt_handler=None):
      rospy.loginfo(f"Starting navigation to named point: '{point_name}'...")
      self.pub_nav_marker.publish(String(data=point_name))
      return self._wait_for_navigation_completion(timeout, interrupt_handler)


  def get_last_reached_marker(self):
      with self.lock:
          return self.last_reached_marker

  def _publish_servo(self, servo_id, angle):
      msg = ServoMove()
      msg.servo_id = int(servo_id)
      msg.angle = int(angle)
      self.pub_servo_control.publish(msg)

  def _publish_base(self, angular_vel):
      msg = NavigationJoyControl()
      msg.angular_velocity = float(angular_vel)
      msg.linear_velocity = 0.0
      self.pub_nav_joy.publish(msg)

  def _rotate_base_to_servo2(self, target_servo2):
      """Delegate to HeadBaseRotator for head-compensated base rotation."""
      try:
          self._rotator.rotate_to_servo2(target_servo2)
          self._face_servo2 = SERVO2_CENTER
      except Exception as e:
          rospy.logerr(f"Base rotation failed: {e}")
          raise

  def face(self, instruction: str) -> bool:
      rospy.loginfo(f"Atomic operation [face]: {instruction}")
      if not instruction:
          instruction = "people"
      
      # Handle directional commands (left, right, up, down, behind) with direct servo/base control
      normalized = instruction.strip().lower()
      if normalized in {"left", "right", "behind", "up", "down"}:
          if normalized == "left":
              base_deg = 70.0
              rospy.loginfo(f"[face] Rotating base left +70.0 deg for '{instruction}'")
              try:
                  self._rotator.rotate_base_by_deg(base_deg, use_feedback=False)
                  self._face_servo2 = SERVO2_CENTER
                  return True
              except Exception as exc:
                  rospy.logerr(f"Face failed: Base rotation error: {exc}")
                  return False
          elif normalized == "right":
              base_deg = -70.0
              rospy.loginfo(f"[face] Rotating base right -70.0 deg for '{instruction}'")
              try:
                  self._rotator.rotate_base_by_deg(base_deg, use_feedback=False)
                  self._face_servo2 = SERVO2_CENTER
                  return True
              except Exception as exc:
                  rospy.logerr(f"Face failed: Base rotation error: {exc}")
                  return False
          elif normalized == "behind":
              base_deg = 180.0
              rospy.loginfo(f"[face] Rotating base behind 180.0 deg for '{instruction}'")
              try:
                  self._rotator.rotate_base_by_deg(base_deg, use_feedback=False)
                  self._face_servo2 = SERVO2_CENTER
                  return True
              except Exception as exc:
                  rospy.logerr(f"Face failed: Base rotation error: {exc}")
                  return False
          elif normalized == "up":
              # Look up: increase servo1 (inverted)
              tilt_deg = -20.0  # Look up 20 degrees (inverted)
              servo1 = SERVO1_CENTER - int(round(tilt_deg * UNITS_PER_DEG))
              if servo1 > SERVO1_MAX:
                  servo1 = SERVO1_MAX
              rospy.loginfo(f"[face] Looking up: servo1={servo1}")
              self._publish_servo(1, servo1)
              self._face_servo1 = servo1
              return True
          elif normalized == "down":
              # Look down: decrease servo1 (inverted)
              tilt_deg = 20.0  # Look down 20 degrees (inverted)
              servo1 = SERVO1_CENTER - int(round(tilt_deg * UNITS_PER_DEG))
              if servo1 < SERVO1_MIN:
                  servo1 = SERVO1_MIN
              rospy.loginfo(f"[face] Looking down: servo1={servo1}")
              self._publish_servo(1, servo1)
              self._face_servo1 = servo1
              return True
      
      # Vision-based face tracking for other targets
      if not self.openai_client:
          rospy.logerr("Face failed: OpenAI client is not initialized.")
          return False

      jpeg_bytes, _ = self.get_rgbd(timeout=3.0)
      if jpeg_bytes is None:
          rospy.logwarn("Face failed: Could not get an image from the camera.")
          return False

      base64_image = base64.b64encode(jpeg_bytes).decode('utf-8')
      
      # Calculate current position in degrees from center
      current_tilt_deg = (self._face_servo1 - SERVO1_CENTER) / UNITS_PER_DEG
      current_pan_deg = (self._face_servo2 - SERVO2_CENTER) / UNITS_PER_DEG
      
      # Calculate angle limits in degrees
      tilt_up_max = (SERVO1_MAX - SERVO1_CENTER) / UNITS_PER_DEG
      tilt_down_min = -(SERVO1_CENTER - SERVO1_MIN) / UNITS_PER_DEG
      pan_left_max = (SERVO2_MAX - SERVO2_CENTER) / UNITS_PER_DEG
      pan_right_min = -(SERVO2_CENTER - SERVO2_MIN) / UNITS_PER_DEG
      
      system_prompt = (
          "You control a robot head with two servos. Output angles in degrees from center (0, 0).\n"
          "Servo1 (tilt): positive=down, negative=up.\n"
          "Servo2 (pan): positive=right, negative=left.\n"
          f"Limits: tilt [{tilt_down_min:.1f} to {tilt_up_max:.1f}], pan [{pan_right_min:.1f} to {pan_left_max:.1f}].\n"
          "Output ONLY JSON: {\"servo1\": 0.0, \"servo2\": 0.0}"
      )
      user_text = (
          f"Instruction: {instruction}\n"
          f"Current angles (degrees): tilt={current_tilt_deg:.1f}, pan={current_pan_deg:.1f}"
      )

      try:
          self._start_busy_audio("analyzing")
          try:
              response = self.openai_client.chat.completions.create(
                  model=self.vqa_model,
                  messages=[
                      {"role": "system", "content": system_prompt},
                      {
                          "role": "user",
                          "content": [
                              {"type": "text", "text": user_text},
                              {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                          ],
                      },
                  ],
                  max_tokens=200,
                  temperature=0,
                  response_format={"type": "json_object"},
              )
          except TypeError:
              response = self.openai_client.chat.completions.create(
                  model=self.vqa_model,
                  messages=[
                      {"role": "system", "content": system_prompt},
                      {
                          "role": "user",
                          "content": [
                              {"type": "text", "text": user_text},
                              {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                          ],
                      },
                  ],
                  max_tokens=200,
                  temperature=0,
              )
          raw = (response.choices[0].message.content or "").strip()
          try:
              data = json.loads(raw)
          except Exception:
              match = re.search(r"\{.*\}", raw, re.DOTALL)
              if match:
                  try:
                      data = json.loads(match.group(0))
                  except Exception:
                      data = None
              else:
                  data = None
              if data is None:
                  preview = raw.replace('\n', ' ')[:300]
                  rospy.logerr(f"Face failed: No valid JSON response. Raw: {preview}")
                  return False
          try:
              tilt_deg = float(data.get("servo1"))
              pan_deg = float(data.get("servo2"))
          except Exception:
              rospy.logerr("Face failed: Invalid angle values.")
              return False
          
          rospy.loginfo(f"[GPT OUTPUT] tilt={tilt_deg:.1f}°, pan={pan_deg:.1f}°")
          
          # Convert degrees to servo values (invert servo1 and servo2)
          servo1 = SERVO1_CENTER - int(round(tilt_deg * UNITS_PER_DEG))
          servo2 = SERVO2_CENTER - int(round(pan_deg * UNITS_PER_DEG))
          
          rospy.loginfo(f"[SERVO VALUES] servo1={servo1} (center-{SERVO1_CENTER-servo1}), servo2={servo2} (center-{SERVO2_CENTER-servo2})")
          
          if servo1 < SERVO1_MIN or servo1 > SERVO1_MAX:
              rospy.logerr(f"Face failed: servo1 out of range ({servo1}, from {tilt_deg:.1f}°).")
              return False
          if servo2 < SERVO2_MIN or servo2 > SERVO2_MAX:
              rospy.logerr(f"Face failed: servo2 out of range ({servo2}, from {pan_deg:.1f}°).")
              return False

          self._publish_servo(1, servo1)
          self._face_servo1 = servo1
          self._face_servo2 = servo2
          try:
              self._rotate_base_to_servo2(servo2)
          except Exception as e:
              # Fallback: convert small servo2 angle to direct base rotation
              rospy.logwarn(f"Head-compensated rotation failed: {e}")
              try:
                  # Calculate angle from servo2 center
                  pan_deg = (SERVO2_CENTER - servo2) / UNITS_PER_DEG
                  if abs(pan_deg) > 1.0:  # Only rotate if angle is significant
                      rospy.loginfo(f"Fallback: Using direct base rotation ({pan_deg:.1f}°)")
                      rotation_time = abs(pan_deg) / math.degrees(BASE_ANG_VEL)
                      ang_vel = BASE_ANG_VEL if pan_deg > 0 else -BASE_ANG_VEL
                      self._publish_base(ang_vel)
                      time.sleep(rotation_time)
                      for _ in range(STOP_BURST_COUNT):
                          self._publish_base(0.0)
                          time.sleep(STOP_BURST_DELAY)
                      rospy.loginfo("Fallback base rotation completed")
                  else:
                      rospy.loginfo("Servo2 angle too small, skipping rotation")
                  self._face_servo2 = SERVO2_CENTER
              except Exception as e2:
                  rospy.logerr(f"Face failed: Fallback rotation error: {e2}")
                  return False
          rospy.loginfo(f"Face applied servo1={self._face_servo1}, servo2={self._face_servo2}")
          return True
      except Exception as e:
          rospy.logerr(f"Face failed: OpenAI call error: {e}")
          return False
      finally:
          self._stop_busy_audio()


  def greet(self) -> bool:
      """Run saved greeting sequence and return to initial position."""
      rospy.loginfo("Running greeting sequence (simulated)")
      return True

  def show(self) -> bool:
      """Run saved show sequence and return to initial position."""
      rospy.loginfo("Running show sequence (simulated)")
      # Placeholder only; no arm movement.
      return True

  def grasp(self, object_name: str) -> bool:
      """Placeholder grasp tool that only narrates an imagined pickup."""
      target = (object_name or "").strip().strip('"').strip("'")
      if not target:
          target = "item"
      self._imagined_grasp_object = target
      rospy.loginfo(f"Running grasp placeholder for '{target}'")
      return self.speak(
          f"If I had arms, I would definitely grab this {target}, "
          "but I still cannot move them, so I am imagining I picked it up."
      )

  def handing(self, object_name: str) -> bool:
      """Placeholder handover tool that only narrates an imagined handover."""
      target = (object_name or "").strip().strip('"').strip("'")
      if not target:
          target = self._imagined_grasp_object or "item"
      rospy.loginfo(f"Running handing placeholder for '{target}'")
      success = self.speak(
          f"If I had arms, I would definitely hand over this {target}, "
          "but I still cannot move it, so I am imagining I handed it over."
      )
      if success:
          self._imagined_grasp_object = None
      return success

  def throwing(self, object_name: str) -> bool:
      """Placeholder throwing tool that only narrates an imagined disposal."""
      target = (object_name or "").strip().strip('"').strip("'")
      if not target:
          target = self._imagined_grasp_object or "item"
      rospy.loginfo(f"Running throwing placeholder for '{target}'")
      success = self.speak(
          f"If I had arms, I would definitely throw this {target} into the bin, "
          "but I still cannot move it, so I am imagining I threw it away."
      )
      if success:
          self._imagined_grasp_object = None
      return success

  def _start_busy_audio(self, audio_name: str):
      if not audio_name:
          return
      if self._busy_audio_thread and self._busy_audio_thread.is_alive():
          return
      self._busy_audio_stop.clear()

      def loop():
          while not self._busy_audio_stop.is_set() and not rospy.is_shutdown():
              self.play_predefined_audio(audio_name)
              if self._busy_audio_stop.wait(PREDEFINED_AUDIO_INTERVAL_SEC):
                  break

      self._busy_audio_thread = threading.Thread(target=loop, daemon=True)
      self._busy_audio_thread.start()

  def _stop_busy_audio(self):
      if self._busy_audio_stop:
          self._busy_audio_stop.set()
      if self._busy_audio_thread and self._busy_audio_thread.is_alive():
          self._busy_audio_thread.join(timeout=1.0)
      self._busy_audio_thread = None

  def consume_nav_assist(self) -> bool:
      if self._nav_assist_played:
          self._nav_assist_played = False
          return True
      return False


  def play_predefined_audio(self, audio_name: str):
      """
        Request a predefined audio scenario by name.
        Args:
            audio_name (str): Scenario name (e.g., 'assist', 'replanning')
        """
      if not audio_name:
          rospy.logwarn("Predefined audio scenario name is empty.")
          return
      try:
          with self._predefined_audio_lock:
              now = time.time()
              next_allowed = self._last_predefined_audio_time + PREDEFINED_AUDIO_INTERVAL_SEC
              if now < next_allowed:
                  time.sleep(next_allowed - now)
              self._last_predefined_audio_time = time.time()
          payload = json.dumps({"s": audio_name})
          self.pub_robot_audio.publish(String(data=payload))
          rospy.loginfo(f"Requested predefined audio scenario: {audio_name}")
      except Exception as e:
          rospy.logerr(f"Failed to request predefined audio '{audio_name}': {e}")


  def _normalize_tts_text(self, text: str) -> str:
      if not text:
          return ""
      normalized = text
      replacements = {
          "\u2014": " - ",  # em dash
          "\u2013": " - ",  # en dash
          "\u2018": "'",    # left single quote
          "\u2019": "'",    # right single quote
          "\u201c": "\"",   # left double quote
          "\u201d": "\"",   # right double quote
          "\u2026": "...",  # ellipsis
          "\u00a0": " ",    # non-breaking space
      }
      for old, new in replacements.items():
          normalized = normalized.replace(old, new)
      normalized = re.sub(r"\s+", " ", normalized).strip()
      return normalized


  def speak(self, text: str) -> bool:
      """Generate speech using Coqui TTS and publish audio to /robot_audio, waiting for playback ack."""
      rospy.loginfo(f"Atomic operation [speak]: Speaking '{text}'")

      if not self.tts_pipeline:
          rospy.logwarn("speak() called but Coqui TTS is not initialized.")
          return False

      if not sf:
          rospy.logerr("soundfile module not available. Cannot save audio.")
          return False

      lock = getattr(self, "_speak_lock", None)
      if lock:
          lock.acquire()
      try:
          # Generate audio using Coqui TTS
          tts_text = self._normalize_tts_text(text)
          if not tts_text:
              rospy.logwarn("speak() called with empty text after normalization.")
              return False
          if tts_text != text:
              rospy.loginfo(f"[TTS] Normalized text: '{tts_text}'")

          rospy.loginfo(f"[TTS] Generating speech audio for: '{tts_text}'")

          import time
          start_time = time.time()

          with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
              audio_file = tmp.name

          # Coqui TTS generates directly to file
          tts_kwargs = {}
          if getattr(self, "tts_speaker", None):
              tts_kwargs["speaker"] = self.tts_speaker
          self.tts_pipeline.tts_to_file(text=tts_text, file_path=audio_file, **tts_kwargs)

          gen_time = time.time() - start_time
          rospy.loginfo(f"[TTS DEBUG] Generation time: {gen_time:.3f}s")

          # Read the generated WAV file
          import soundfile
          audio_np, sample_rate = soundfile.read(audio_file)

          # Ensure audio is 1D array
          if len(audio_np.shape) > 1:
              audio_np = audio_np.flatten()

          # Save as WAV using soundfile (16-bit PCM)
          sf.write(audio_file, audio_np, sample_rate, subtype='PCM_16')
          rospy.loginfo(f"Audio generated ({len(audio_np)} samples @ {sample_rate}Hz)")

          # Read audio file as bytes
          with open(audio_file, 'rb') as f:
              audio_bytes = f.read()

          # Publish audio bytes to /robot_audio topic as base64 string with an id
          import base64
          audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
          audio_id = uuid.uuid4().hex
          payload = json.dumps({"id": audio_id, "audio_b64": audio_b64})

          done_evt = threading.Event()
          with self._audio_done_lock:
              self._audio_done_events[audio_id] = done_evt

          self.pub_robot_audio.publish(String(data=payload))
          rospy.loginfo(f"Published audio data to /robot_audio ({len(audio_bytes)} bytes), id={audio_id}")

          # Wait for playback acknowledgement so navigation doesn't overlap speech
          if not done_evt.wait(timeout=60.0):
              rospy.logwarn(f"[AUDIO] Timeout waiting for playback ack (id={audio_id}). Continuing.")
          with self._audio_done_lock:
              self._audio_done_events.pop(audio_id, None)
          return True
      except Exception as e:
          rospy.logerr(f"Failed to generate or publish speech: {e}")
          return False
      finally:
          if lock:
              lock.release()


  def vqa(self, question: str) -> str:
      rospy.loginfo(f"Atomic operation [vqa]: Answering question '{question}'")


      if not self.openai_client:
          rospy.logerr("VQA failed: OpenAI client is not initialized.")
          return "I can't answer questions right now due to a configuration error."


      try:
          self._start_busy_audio("analyzing")
          rospy.sleep(3.0)
          jpeg_bytes, _ = self.get_rgbd(timeout=3.0)
          if jpeg_bytes is None:
              rospy.logwarn("VQA failed: Could not get an image from the camera.")
              return "I couldn't see anything, so I can't answer the question."

          base64_image = base64.b64encode(jpeg_bytes).decode('utf-8')
          rospy.loginfo("Sending request to OpenAI Vision API...")
          response = self.openai_client.chat.completions.create(
              model=self.vqa_model,
              messages=[
                  {
                      "role": "user",
                      "content": [
                          {"type": "text", "text": question},
                          {
                              "type": "image_url",
                              "image_url": {
                                  "url": f"data:image/jpeg;base64,{base64_image}"
                              }
                          }
                      ]
                  }
              ],
              max_tokens=300
          )
          answer = response.choices[0].message.content
          rospy.loginfo(f"VQA model returned: '{answer}'")
          return answer
      except Exception as e:
          rospy.logerr(f"Failed to call OpenAI VQA API: {e}")
          return "Sorry, I encountered an error while trying to analyze the scene."
      finally:
          self._stop_busy_audio()



class LLMPlanner:
  """
  A planner class that generates a task plan by interacting with an LLM,
  parses the plan, and executes it sequentially using the RobotAgent.
  """
  def __init__(self, agent, current_location, scene_graph_path, openai_model="gpt-5.2"):
      self.agent = agent
      self.scene_graph = self._load_scene_graph(scene_graph_path)
      self.current_location = _load_persisted_location(current_location)
      valid_locations = {
          node.get("node_name")
          for node in self.scene_graph
          if isinstance(node, dict) and node.get("node_name")
      }
      if valid_locations and self.current_location not in valid_locations:
          rospy.logwarn(
              f"[INIT] Persisted current_location '{self.current_location}' is not in "
              f"{scene_graph_path}; resetting to '{current_location}'."
          )
          self.current_location = current_location
      _save_persisted_location(self.current_location)
      self.sim_scene_nodes = None
      try:
          self.sim_scene_nodes = load_scene_graph(scene_graph_path)
      except Exception as e:
          rospy.logwarn(f"Simulator validation disabled: {e}")

      self.openai_client = OpenAI()
      self.openai_model = openai_model
      self._supports_responses = hasattr(self.openai_client, "responses")
      self._anchor_response_id = None
      if self._supports_responses:
          self._anchor_response_id = _get_anchor_response_id(
              self.openai_client, scene_graph_path, self.openai_model
          )
      if self._anchor_response_id:
          self.scene_graph_str = "(scene graph provided earlier)"
      else:
          self.scene_graph_str = read_json_as_str(scene_graph_path)
      self._session_response_id = self._anchor_response_id
      self._session_chat_messages = []
      self._llm_lock = threading.Lock()

      # Try seeding initial logical location from the robot's pose using map_new_markers.json.
      # This only runs once at planner initialization: if the agent already has a pose
      # and the provided current_location looks like a default (or is empty), we
      # map the pose to the nearest marker and use that as the initial location.
      # try:
      #     seed_timeout = 2.0
      #     start_t = time.time()
      #     while time.time() - start_t < seed_timeout:
      #         with getattr(self.agent, 'lock', threading.Lock()):
      #             pose = getattr(self.agent, 'current_pose', None)
      #         if pose:
      #             # Only overwrite obvious defaults (common default 'charging_point' or empty)
      #             if not self.current_location or self.current_location == 'charging_point':
      #                 nearest = self.agent.get_nearest_marker_from_map()
      #                 if nearest:
      #                     self.current_location = nearest
      #                     rospy.loginfo(f"[INIT] Seeded initial location from pose: {self.current_location}")
      #             break
      #         time.sleep(0.1)
      # except Exception as e:
      #     rospy.logwarn(f"Failed to seed initial location from pose: {e}")

      # For parallel task execution
      self.task_thread = None
      self.task_running = False

      self.pub_instruction = rospy.Publisher('/agent/instruction', String, queue_size=10, latch=True)
      self.pub_plan = rospy.Publisher('/agent/plan', String, queue_size=10, latch=True)
      self.pub_current_step = rospy.Publisher('/agent/current_step', String, queue_size=10, latch=True)
      self.pub_vqa_answer = rospy.Publisher('/agent/vqa_answer', String, queue_size=10, latch=True)
      self.pub_interrupt_ack = rospy.Publisher('/interrupt_ack', String, queue_size=10)
      self.sub_stop = rospy.Subscriber('/agent/stop', String, self._stop_callback)

      # Interrupt handling
      self.interrupt_enabled = True
      self.interrupt_queue = []
      self.interrupt_lock = threading.Lock()
      self.voice_question_listener_thread = None
      self.last_question_time = 0.0
      self.mic_blocked = False
      self._planning_notice_stop = threading.Event()
      self._planning_notice_thread = None
      self._pause_requested = False
      self._abort_requested = False
      self._continue_requested = False
      self._pause_reason = ""
      self.plan_ready_phrases = [
          "Plan ready. Moving now.",
          "Got a plan. Heading out.",
          "Plan received. Let's go.",
          "All set. Starting now.",
      ]
      self._log_prefix = "[AGENT]"
      self._last_interrupt_text = None
      self._last_interrupt_time = 0.0
      self._interrupt_dedupe_seconds = 2.0
      self.conversation_history = []

  # Basic plan execution with a stop button; no other external control actions.


  def _log_section(self, title: str):
      line = "=" * 60
      rospy.loginfo(line)
      rospy.loginfo(f"{self._log_prefix} {title}")
      rospy.loginfo(line)

  def _stop_callback(self, msg):
      if self._abort_requested:
          return
      self._abort_requested = True
      self._pause_requested = False
      self._continue_requested = False
      self._pause_reason = "stop"
      rospy.logwarn("[STOP] Stop requested. Cancelling navigation.")
      self.agent.cancel_nav()
      self.agent.speak("I'll stop.")
      # Update current_location to the last reached location
      reached = self.agent.get_last_reached_marker()
      if reached:
          self._set_current_location(reached)
          rospy.loginfo(f"[STOP] Updated current_location to: {self.current_location}")


  def _should_dedupe_question(self, text: str) -> bool:
      if not text:
          return True
      now = time.time()
      if self._last_interrupt_text == text and (now - self._last_interrupt_time) < self._interrupt_dedupe_seconds:
          return True
      self._last_interrupt_text = text
      self._last_interrupt_time = now
      return False

  def _last_assistant_was_question(self) -> bool:
      if not self.conversation_history:
          return False
      last = self.conversation_history[-1]
      if last.get("role") != "assistant":
          return False
      content = (last.get("content") or "").strip()
      if not content:
          return False
      for line in content.splitlines():
          line = line.strip()
          if line.startswith("speak(") and "?" in line:
              return True
      return False

  def _load_scene_graph(self, path):
      with open(path, 'r') as f:
          return json.load(f)

  def _set_current_location(self, location):
      if not location:
          return
      self.current_location = location
      _save_persisted_location(location)

  def _start_instruction_session(self):
      with self._llm_lock:
          self._session_response_id = self._anchor_response_id
          self._session_chat_messages = []

  def _reset_instruction_session(self):
      with self._llm_lock:
          self._session_response_id = self._anchor_response_id
          self._session_chat_messages = []

  def _extract_response_text(self, response):
      text = getattr(response, "output_text", None)
      if text:
          return text
      try:
          return response.output[0].content[0].text
      except Exception:
          return ""


  def _query_llm(
      self,
      prompt,
      previous_response_id=None,
      temperature=0.0,
      max_output_tokens=None,
      timeout=None,
  ):
      with self._llm_lock:
          if self._supports_responses:
              if previous_response_id is None:
                  previous_response_id = self._session_response_id
              kwargs = {
                  "model": self.openai_model,
                  "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                  "temperature": temperature,
              }
              if previous_response_id:
                  kwargs["previous_response_id"] = previous_response_id
              if max_output_tokens is not None:
                  kwargs["max_output_tokens"] = max_output_tokens
              if timeout is not None:
                  kwargs["timeout"] = timeout
              response = self.openai_client.responses.create(**kwargs)
              text = self._extract_response_text(response).strip()
              response_id = getattr(response, "id", None)
              if response_id:
                  self._session_response_id = response_id
              return text, response_id

          if self._session_chat_messages is None:
              self._session_chat_messages = []
          self._session_chat_messages.append({"role": "user", "content": prompt})
          kwargs = {
              "model": self.openai_model,
              "messages": self._session_chat_messages,
              "temperature": temperature,
          }
          if max_output_tokens is not None:
              kwargs["max_tokens"] = max_output_tokens
          if timeout is not None:
              kwargs["timeout"] = timeout
          response = self.openai_client.chat.completions.create(**kwargs)
          text = response.choices[0].message.content.strip()
          self._session_chat_messages.append({"role": "assistant", "content": text})
          return text, None


  def _parse_plan(self, plan_text):
      plan = []
      for raw_line in plan_text.splitlines():
          line = raw_line.strip()
          if not line:
              continue
          match = _PLAN_LINE_RE.match(line)
          if not match:
              continue
          action, args = match.groups()
          args = args.strip().strip('"').strip("'") if args else None
          plan.append({"action": action, "args": args})
      return plan


  def _bfs_shortest(self, start, goal, graph, markers=None, junction_only=False):
      if start not in graph or goal not in graph:
          return None
      if start == goal:
          return []

      q = deque([[start]])
      visited = {start}

      def _allowed(node):
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
          for nxt in graph.get(node, []):
              if nxt in visited:
                  continue
              if not _allowed(nxt):
                  continue
              visited.add(nxt)
              q.append(path + [nxt])

      return None

  def _build_expected_paths_for_retry(self, plan_text):
      if not self.sim_scene_nodes or not plan_text:
          return []

      name_map = {name.lower(): name for name in self.sim_scene_nodes.keys()}

      def _resolve(name):
          if name is None:
              return None
          return name_map.get(str(name).strip().lower())

      graph = {
          name: list(info.get("neighbors") or [])
          for name, info in self.sim_scene_nodes.items()
      }
      markers = {name: info.get("marker", "S") for name, info in self.sim_scene_nodes.items()}

      segments = []
      current_path = []
      for raw_line in plan_text.splitlines():
          line = raw_line.strip()
          if not line:
              continue
          match = _PLAN_LINE_RE.match(line)
          if not match:
              continue
          action, args = match.groups()
          action = action.strip().lower()
          if action not in {"pass", "nav"}:
              continue
          if not args:
              continue
          arg = args.strip().strip('"').strip("'")
          if not arg:
              continue
          current_path.append(arg)
          if action == "nav":
              segments.append(list(current_path))
              current_path = []

      if not segments:
          return []

      current = _resolve(self.current_location) or self.current_location
      expected_paths = []
      for idx, seg in enumerate(segments, start=1):
          dest_raw = seg[-1]
          dest = _resolve(dest_raw)
          entry = {
              "segment": idx,
              "segment_start": current,
              "segment_destination": dest or dest_raw,
          }
          if dest is None:
              entry["expected_path"] = None
              entry["expected_path_type"] = "none"
              entry["error"] = f"Node does not exist: {dest_raw}"
              expected_paths.append(entry)
              continue

          expected_path = self._bfs_shortest(
              current, dest, graph, markers=markers, junction_only=True
          )
          expected_path_type = "o_only"
          if expected_path is None:
              expected_path = self._bfs_shortest(
                  current, dest, graph, markers=markers, junction_only=False
              )
              expected_path_type = "fallback" if expected_path is not None else "none"
          entry["expected_path"] = expected_path
          entry["expected_path_type"] = expected_path_type
          expected_paths.append(entry)
          current = dest

      return expected_paths

  def _format_expected_paths(self, expected_paths):
      if not expected_paths:
          return ""
      lines = ["Expected shortest paths per segment:"]
      for entry in expected_paths:
          segment = entry.get("segment")
          start = entry.get("segment_start")
          dest = entry.get("segment_destination")
          error = entry.get("error")
          if error:
              lines.append(f"{segment}) {start} -> {dest}: {error}")
              continue
          path = entry.get("expected_path")
          if path is None:
              path_str = "None"
          elif not path:
              path_str = "[]"
          else:
              path_str = " -> ".join(path)
          path_type = entry.get("expected_path_type", "unknown")
          lines.append(f"{segment}) {start} -> {dest}: {path_str} ({path_type})")
      return "\n".join(lines)

  def _build_retry_prompt_with_paths(self, reason, expected_paths):
      reason_text = reason or "Unknown error"
      expected_block = self._format_expected_paths(expected_paths)
      if expected_block:
          return (
              "The plan is invalid.\n"
              f"Reason: {reason_text}.\n"
              f"{expected_block}\n"
              "Fix the plan.\n"
              "Output only tool calls."
          )
      return build_retry_prompt(reason_text)

  def _is_interrupt_question_text(self, text: str) -> bool:
      if not text:
          return False
      return bool(text.strip()) and self.interrupt_enabled


  def _process_interrupt_queue(self):
      while True:
          with self.interrupt_lock:
              if not self.interrupt_queue:
                  return
              question_data = self.interrupt_queue.pop(0)

          question = question_data.get("question", "")
          question_id = question_data.get("question_id", "")
          if not question:
              continue
          self._handle_interrupt_question(question, question_id)


  def _handle_interrupt_question(self, question: str, question_id: str = ""):
      try:
          self.pub_interrupt_ack.publish(f"Processing: {question}")
          rospy.loginfo(f"[QNA] Question: '{question}'")

          prompt = (
              "You are answering a question during robot operation. Provide ONLY a factual descriptive answer in plain conversational text.\n"
              "If the question mentions a location/object in the scene graph below, describe what's there based on the scene graph.\n"
              "If unrelated to the scene graph, answer from general knowledge.\n"
              "CRITICAL: Output plain text ONLY - NO tool calls, NO nav() commands, NO function syntax, NO code.\n"
              "Max 25 words, direct and conversational.\n"
              f"Current location: {self.current_location}\n"
              f"Scene graph:\n{self.scene_graph_str}\n"
              f"Question: {question}\n"
              "Answer:"
          )
          answer, _ = self._query_llm(
              prompt,
              temperature=0.2,
              max_output_tokens=80,
              timeout=8,
          )
          answer = (answer or "").strip()
          rospy.loginfo(f"[QNA] Answer: '{answer}'")
          self.agent.speak(answer)
          self.last_question_time = time.time()
          return True
      except Exception as e:
          rospy.logerr(f"[Interrupt] Error handling question: {e}")
          error_msg = "Sorry, I couldn't process that question right now."
          self.agent.speak(error_msg)
          return False


  def _start_voice_question_listener(self):
      if self.voice_question_listener_thread is not None:
          return

      def voice_listener_loop():
          rospy.loginfo("[Interrupt] Voice listener thread started")
          while self.interrupt_enabled and not rospy.is_shutdown():
              try:
                  if self.mic_blocked:
                      time.sleep(0.1)
                      continue
                  if not is_audio_available():
                      time.sleep(0.2)
                      continue
                  text = get_audio_text()
                  if not text:
                      time.sleep(0.05)
                      continue
                  if self._is_interrupt_question_text(text):
                      if self._should_dedupe_question(text):
                          time.sleep(0.05)
                          continue
                      self.pub_interrupt_ack.publish(f"Received voice question: {text}")
                      with self.interrupt_lock:
                          self.interrupt_queue.append({
                              "question": text,
                              "question_id": "voice"
                          })
                      self.last_question_time = time.time()
                      rospy.loginfo(f"[QNA] Queued: '{text}'")
                      self._process_interrupt_queue()
                  time.sleep(0.05)
              except Exception as e:
                  rospy.logerr(f"[Interrupt] Voice listener error: {e}")
                  time.sleep(0.2)
          rospy.loginfo("[Interrupt] Voice listener thread stopped")

      self.voice_question_listener_thread = threading.Thread(target=voice_listener_loop, daemon=True)
      self.voice_question_listener_thread.start()


  def _stop_voice_question_listener(self):
      if self.voice_question_listener_thread is not None:
          self.voice_question_listener_thread.join(timeout=2.0)
          self.voice_question_listener_thread = None

  def _start_planning_notices(self):
      if self._planning_notice_thread and self._planning_notice_thread.is_alive():
          return
      self._planning_notice_stop.clear()
      self.agent.play_predefined_audio('wait')

      def loop():
          while not self._planning_notice_stop.is_set() and not rospy.is_shutdown():
              if self._planning_notice_stop.wait(PREDEFINED_AUDIO_INTERVAL_SEC):
                  break
              self.agent.play_predefined_audio('replanning')

      self._planning_notice_thread = threading.Thread(target=loop, daemon=True)
      self._planning_notice_thread.start()

  def _stop_planning_notices(self):
      if self._planning_notice_stop:
          self._planning_notice_stop.set()
      if self._planning_notice_thread and self._planning_notice_thread.is_alive():
          self._planning_notice_thread.join(timeout=1.0)
      self._planning_notice_thread = None

  def _get_nav_failure_assist_delay_seconds(self) -> float:
      env_value = os.environ.get("NAV_FAILURE_ASSIST_DELAY_SECONDS")
      if env_value:
          try:
              default_value = float(env_value)
          except ValueError:
              default_value = NAV_FAILURE_ASSIST_DELAY_SECONDS_DEFAULT
      else:
          default_value = NAV_FAILURE_ASSIST_DELAY_SECONDS_DEFAULT
      try:
          return float(rospy.get_param("~nav_failure_assist_delay_seconds", default_value))
      except Exception:
          return default_value

  def _speak_plan_ready(self):
      return

  def _build_completion_phrase(self, success: bool) -> str:
      if success:
          prompt = (
              "You are a robot. Generate one short, natural completion sentence. "
              "Use ASCII only. Keep it under 8 words. No quotes. Success=true."
          )
          fallback = "All done."
      else:
          prompt = (
              "You are a robot. Generate one short, natural failure sentence that asks for help. "
              "Use ASCII only. Keep it under 12 words. No quotes. Success=false."
          )
          fallback = "I couldn't finish that. Please assist me."
      return self._generate_phrase(prompt, fallback)

  def _generate_phrase(self, prompt: str, fallback: str) -> str:
      try:
          response = self.openai_client.chat.completions.create(
              model=self.openai_model,
              messages=[{"role": "user", "content": prompt}],
              temperature=0.6,
              max_tokens=30,
          )
          text = response.choices[0].message.content.strip()
          text = text.strip('"').strip("'").replace("\n", " ").strip()
          text = text.encode("ascii", "ignore").decode()
          if not text:
              return fallback
          return text
      except Exception:
          return fallback

  def _humanize_node_name(self, node_name: str) -> str:
      if not node_name:
          return "the checked area"
      return str(node_name).replace("_", " ").strip()

  def _build_vqa_summary_speech(self, vqa_results: list[dict]) -> str:
      if not vqa_results:
          return ""

      lines = []
      for item in vqa_results:
          location = self._humanize_node_name(item.get("location"))
          answer = str(item.get("answer") or "").strip()
          answer = " ".join(answer.split())
          if not answer:
              answer = "I could not get a clear answer."
          lines.append(f"At {location}: {answer}")

      if len(lines) == 1:
          return f"Here is what I found. {lines[0]}"
      return "Here is what I found. " + " ".join(lines)


  # Removed: _apply_control_action (no external control)


  def _write_status(self, state: str):
      try:
          data = {
              "state": state,
              "timestamp": datetime.now().isoformat(),
              "available": True
          }
          with open(STATUS_FILE, 'w') as f:
              json.dump(data, f, indent=2)
      except Exception:
          pass


  # Removed: _read_control_action (no external control)


  # Removed: _start_control_listener (no external control)


  # Removed: _stop_control_listener (no external control)


  # Removed: _wait_for_resume_or_abort (no external control)


  # Removed: _handle_pause (no external control)


  def execute_task(self, instruction, succeeded_steps=None, failed_step=None):
      self._log_section(f"TASK START: {instruction}")

      self.last_question_time = 0.0
      self.mic_blocked = False
      self.interrupt_enabled = True
      self._pause_requested = False
      self._abort_requested = False
      self._continue_requested = False
      self._pause_reason = ""
      self._write_status("running")

      if self.conversation_history:
          classify_prompt = classify_topic_change(instruction, self.conversation_history)
          try:
              classification_text, _ = self._query_llm(
                  classify_prompt,
                  temperature=0.3,
                  max_output_tokens=32,
              )
              if classification_text and "new_topic" in classification_text.lower():
                  rospy.loginfo("[DEBUG] Detected NEW TOPIC - clearing conversation history")
                  self.conversation_history = []
                  self._reset_instruction_session()
              else:
                  rospy.loginfo("[DEBUG] Detected CONTINUATION - keeping conversation history")
          except Exception as e:
              rospy.logwarn(f"[DEBUG] Topic classification error: {e}. Assuming continuation.")

      self.conversation_history.append({"role": "user", "content": instruction})

      self._start_voice_question_listener()
      if not self._last_assistant_was_question():
          self._start_planning_notices()

      try:
          self.pub_instruction.publish(instruction)
          self.pub_plan.publish("")
          self.pub_current_step.publish("-1")
          self.pub_vqa_answer.publish("")

          # If replanning, inform LLM of previous progress
          prompt = generate_agent_prompt(
              current_location=self.current_location,
              scene_graph_str=self.scene_graph_str,
              instruction=instruction,
              conversation_history=self.conversation_history,
              held_object=getattr(self.agent, "_imagined_grasp_object", None),
          )
          plan_context = ""
          if succeeded_steps is not None and failed_step is not None:
              plan_context = (
                  "The previous plan failed at this step: "
                  f"{failed_step['action']}({failed_step['args']})\n"
                  "The following steps were already completed successfully (do not repeat them):\n" +
                  "\n".join([f"{s['action']}({s['args']})" for s in succeeded_steps]) +
                  "\nPlease generate a new plan to fulfill the original instruction, starting from where it left off."
              )
          if plan_context:
              prompt += "\n\n" + plan_context

          max_attempts = 3
          retry_reason = None
          retry_expected_paths = None
          plan_text = None
          validation_result = None
          previous_response_id = self._anchor_response_id

          for attempt in range(1, max_attempts + 1):
              rospy.loginfo(f"[PLAN] Querying LLM (attempt {attempt}/{max_attempts})...")
              llm_prompt = prompt
              if retry_reason:
                  llm_prompt = prompt + "\n\n" + self._build_retry_prompt_with_paths(
                      retry_reason,
                      retry_expected_paths,
                  )
              plan_text, response_id = self._query_llm(llm_prompt, previous_response_id)
              if response_id:
                  previous_response_id = response_id
              rospy.loginfo(f"[PLAN] LLM returned plan:\n{plan_text}")

              if self.sim_scene_nodes is None:
                  break

              validation_result = run_simulator(
                  gpt_output=plan_text,
                  current_location=self.current_location,
                  scene_nodes=self.sim_scene_nodes,
              )
              if validation_result.get("status") == "VALID":
                  break

              retry_reason = _format_validation_failure(validation_result)
              retry_expected_paths = self._build_expected_paths_for_retry(plan_text)
              rospy.logwarn(f"[PLAN] Invalid: {retry_reason}")
              expected_path = validation_result.get("expected_path")
              if expected_path is not None:
                  expected_type = validation_result.get("expected_path_type", "unknown")
                  rospy.logwarn(f"Expected path ({expected_type}): {expected_path}")
              given_path = validation_result.get("given_path")
              if given_path is not None:
                  rospy.logwarn(f"Given path: {given_path}")
              segment = validation_result.get("segment")
              if segment is not None:
                  seg_start = validation_result.get("segment_start")
                  seg_dest = validation_result.get("segment_destination")
                  rospy.logwarn(f"Segment {segment} start={seg_start} destination={seg_dest}")
              if attempt < max_attempts:
                  rospy.loginfo("[PLAN] Replanning with simulator feedback...")
                  continue
              rospy.logerr("Failed to generate a valid plan after retries.")
              self._stop_planning_notices()
              self.agent.play_predefined_audio('assist')
              return

          self._stop_planning_notices()

          self._log_section("PLAN RECEIVED")
          plan = self._parse_plan(plan_text)
          if not plan:
              rospy.logerr("LLM did not generate any executable operations. Please check the instruction and scene graph.")
              return

          rospy.loginfo(f"[PLAN] Parsed {len(plan)} steps:")
          for i, step in enumerate(plan):
              rospy.loginfo(f"[PLAN]   Step {i+1}: {step['action']}({step['args'] or ''})")

          final_nav_target = None
          for step in plan:
              if step["action"] == "nav" and step["args"]:
                  final_nav_target = step["args"]

          plan_str_list = [f"{step['action']}({step['args'] or ''})" for step in plan]
          full_plan_str = "\n".join(plan_str_list)
          self.pub_plan.publish(full_plan_str)
          self._speak_plan_ready()

          self._log_section("EXECUTION START")
          success = False
          pending_passes = []
          pending_speaks = []
          plan_has_nav = any(step["action"] == "nav" for step in plan)
          nav_completed = not plan_has_nav
          index = 0
          succeeded_steps = [] if succeeded_steps is None else list(succeeded_steps)
          origin_location = self.current_location
          vqa_results = []
          vqa_started = False

          def flush_pending_speaks():
              nonlocal pending_speaks
              if not pending_speaks:
                  return
              for text in pending_speaks:
                  rospy.loginfo(f"[SPEAK] Robot speaking: '{text}'")
                  self.agent.speak(text)
                  if self.interrupt_enabled:
                      self._process_interrupt_queue()
              pending_speaks = []

          def interrupt_handler():
              if self.interrupt_enabled:
                  self._process_interrupt_queue()

          def make_nav_progress_handler(marker_steps):
              last_marker = {"name": None}
              reported_positions = set()

              def handler():
                  interrupt_handler()
                  reached = self.agent.get_last_reached_marker()
                  if not reached or reached == last_marker["name"]:
                      return
                  last_marker["name"] = reached
                  for pos, (marker_name, step_index) in enumerate(marker_steps):
                      if pos in reported_positions:
                          continue
                      if marker_name == reached:
                          self.pub_current_step.publish(str(step_index))
                          reported_positions.add(pos)
                          break

              return handler

          def trim_markers_after_reached(markers, marker_steps, reached):
              if not reached:
                  return markers, marker_steps
              if reached not in markers:
                  return markers, marker_steps
              idx = markers.index(reached)
              return markers[idx + 1:], marker_steps[idx + 1:]

          def request_new_instruction(destination):
              dest = destination or "the destination"
              self.agent.speak(f"I cannot go to {dest}. Where do you want to go instead?")
              new_instruction = get_audio_instruction("> Instruction: ")
              if not new_instruction:
                  return False
              if is_exit_instruction(new_instruction):
                  cleanup_transient_json_state()
                  rospy.loginfo("Exit command received. Closing agent_cv.py.")
                  rospy.signal_shutdown("Exit command received.")
                  return False
              rospy.loginfo(f"[VOICE] Instruction: {new_instruction}")
              return self.execute_task(new_instruction)

          def attempt_nav_with_retries(markers, marker_steps, final_destination):
              remaining_markers = list(markers)
              remaining_steps = list(marker_steps)
              failures = 0
              while True:
                  if self._abort_requested:
                      return False
                  nav_interrupt_handler = make_nav_progress_handler(remaining_steps)
                  if len(remaining_markers) > 1:
                      success = self.agent.nav_multipoint(
                          remaining_markers, interrupt_handler=nav_interrupt_handler
                      )
                  else:
                      success = self.agent.nav(
                          remaining_markers[0], interrupt_handler=nav_interrupt_handler
                      )
                  if self._abort_requested:
                      return False
                  if success:
                      reached = self.agent.get_last_reached_marker()
                      self._set_current_location(reached or remaining_markers[-1])
                      return True
                  reached = self.agent.get_last_reached_marker()
                  if reached:
                      self._set_current_location(reached)
                      if reached == remaining_markers[-1]:
                          return True
                  failures += 1
                  if failures >= 3:
                      return request_new_instruction(final_destination)
                  if not self.agent.consume_nav_assist():
                      self.agent.play_predefined_audio('assist')
                  time.sleep(NAV_ASSIST_WAIT_SECONDS)
                  if self._abort_requested:
                      return False
                  remaining_markers, remaining_steps = trim_markers_after_reached(
                      remaining_markers, remaining_steps, reached
                  )
                  if not remaining_markers:
                      return True

          def run_pending_passes_before(action_name):
              nonlocal pending_passes, nav_completed
              if not pending_passes:
                  return True
              rospy.logwarn(f"Pending pass nodes found before {action_name}. Executing multipoint to last pass.")
              nav_completed = False
              last_pending = pending_passes[-1][0]
              markers = [name for name, _ in pending_passes]
              marker_steps = list(pending_passes)
              nav_interrupt_handler = make_nav_progress_handler(marker_steps)
              success = self.agent.nav_multipoint(markers, interrupt_handler=nav_interrupt_handler)
              if success:
                  reached = self.agent.get_last_reached_marker()
                  self._set_current_location(reached or last_pending)
                  nav_completed = True
                  flush_pending_speaks()
              pending_passes = []
              if not success:
                  reached = self.agent.get_last_reached_marker()
                  if reached:
                      self._set_current_location(reached)
                  rospy.logerr(f"Failed while executing pending pass nodes before {action_name}.")
                  self.pub_current_step.publish("-2")
                  return False
              return True

          # Main plan execution loop: executes each step in order, no external control
          while index < len(plan):
              if self._abort_requested:
                  rospy.logwarn("[STOP] Task aborted by stop request.")
                  self._write_status("idle")
                  self.pub_current_step.publish("-2")
                  return
              self._process_interrupt_queue()
              step = plan[index]
              rospy.loginfo(f"[EXEC] Step {index+1}/{len(plan)}: {step['action']}({step['args'] or ''})")

              self.pub_current_step.publish(str(index))

              action = step["action"]
              args = step["args"]
              success = False

              if action == "pass":
                  if not args:
                      rospy.logwarn("pass operation missing location parameter.")
                      break
                  pending_passes.append((args, index))
                  success = True
                  succeeded_steps.append(step)
                  index += 1
                  continue

              if action == "nav":
                  if not args:
                      rospy.logwarn("nav operation missing location parameter.")
                      break
                  nav_completed = False
                  if pending_passes:
                      markers = [name for name, _ in pending_passes] + [args]
                      marker_steps = list(pending_passes) + [(args, index)]
                      pending_passes = []
                      result = attempt_nav_with_retries(
                          markers, marker_steps, final_nav_target or args
                      )
                  else:
                      markers = [args]
                      marker_steps = [(args, index)]
                      result = attempt_nav_with_retries(
                          markers, marker_steps, final_nav_target or args
                      )
                  if result is True:
                      nav_completed = True
                      flush_pending_speaks()
                      succeeded_steps.append(step)
                      index += 1
                      continue
                  if self._abort_requested:
                      self._write_status("idle")
                  return
              # ...existing code for other actions...
              if action == "face":
                  if not args:
                      rospy.logwarn("face operation missing target parameter.")
                      break
                  success = self.agent.face(args)
                  if success:
                      succeeded_steps.append(step)
              elif action == "speak":
                  if vqa_started:
                      rospy.loginfo("[VQA] Skipping planned speak after vqa(); will report from actual VQA results.")
                      success = True
                      succeeded_steps.append(step)
                  else:
                      if not args:
                          rospy.logwarn("speak operation missing text parameter.")
                          break
                      args = args.replace("\\n", "\n")
                      success = self.agent.speak(args)
                      if self.interrupt_enabled:
                          self._process_interrupt_queue()
                      if success:
                          succeeded_steps.append(step)
              elif action == "greet":
                  success = self.agent.greet()
                  if success:
                      succeeded_steps.append(step)
              elif action == "grasp":
                  if not args:
                      rospy.logwarn("grasp operation missing object parameter.")
                      break
                  success = self.agent.grasp(args)
                  if success:
                      succeeded_steps.append(step)
              elif action == "handing":
                  if not args:
                      rospy.logwarn("handing operation missing object parameter.")
                      break
                  success = self.agent.handing(args)
                  if success:
                      succeeded_steps.append(step)
              elif action == "throwing":
                  if not args:
                      rospy.logwarn("throwing operation missing object parameter.")
                      break
                  success = self.agent.throwing(args)
                  if success:
                      succeeded_steps.append(step)
              elif action == "vqa":
                  if not run_pending_passes_before("vqa"):
                      break
                  if not args:
                      rospy.logwarn("vqa operation missing question parameter.")
                      break
                  vqa_started = True
                  answer = self.agent.vqa(args)
                  rospy.loginfo(f"VQA result: {answer}")
                  self.pub_vqa_answer.publish(answer)
                  vqa_results.append(
                      {
                          "location": self.current_location,
                          "question": args,
                          "answer": answer,
                      }
                  )
                  success = True
                  if success:
                      succeeded_steps.append(step)
              else:
                  rospy.logerr(f"Unknown operation: {action}")
                  break

              if not success:
                  rospy.logerr(f"Step {action} execution failed! Task aborted.")
                  self.pub_current_step.publish("-2")
                  break

              index += 1

          if self._abort_requested:
              rospy.logwarn("[STOP] Task aborted by stop request.")
              self._write_status("idle")
              return

          completed = index >= len(plan) and not self._abort_requested and not self._pause_requested
          if pending_passes and completed:
              rospy.logwarn("[EXEC] Plan ended with pass nodes only. Executing multipoint to last pass.")
              nav_completed = False
              markers = [name for name, _ in pending_passes]
              marker_steps = list(pending_passes)
              nav_interrupt_handler = make_nav_progress_handler(marker_steps)
              success = self.agent.nav_multipoint(markers, interrupt_handler=nav_interrupt_handler)
              if success:
                  reached = self.agent.get_last_reached_marker()
                  self._set_current_location(reached or pending_passes[-1][0])
                  nav_completed = True
                  flush_pending_speaks()
              else:
                  reached = self.agent.get_last_reached_marker()
                  if reached:
                      self._set_current_location(reached)
                  self.pub_current_step.publish("-2")
          if completed and vqa_results and not self._abort_requested:
              if origin_location and self.current_location != origin_location:
                  rospy.loginfo(f"[VQA] Returning to original location for report: {origin_location}")
                  returned = attempt_nav_with_retries(
                      [origin_location],
                      [(origin_location, len(plan))],
                      origin_location,
                  )
                  if returned is not True:
                      rospy.logerr("[VQA] Failed to return to original location for reporting.")
                      success = False
                  else:
                      nav_completed = True
              if success:
                  summary = self._build_vqa_summary_speech(vqa_results)
                  if summary:
                      spoke = self.agent.speak(summary)
                      if not spoke:
                          rospy.logwarn("[VQA] Failed to speak VQA summary.")
          if success:
              rospy.loginfo("[EXEC] Task execution flow completed successfully.")
              self.pub_current_step.publish("-3")
          else:
              rospy.logerr("[EXEC] Task execution failed.")
              self.agent.speak(self._build_completion_phrase(False))
          self._write_status("idle")
          self._log_section("TASK END")
          if plan_text:
              self.conversation_history.append({"role": "assistant", "content": plan_text})
      finally:
          self._stop_planning_notices()
          if self.interrupt_enabled:
              self._process_interrupt_queue()
          self.interrupt_enabled = False
          self._stop_voice_question_listener()



if __name__ == '__main__':
  print("\n" + "="*60)
  print("VOICE INPUT MODE ACTIVE")
  print("Listening for phone instructions...")
  print("="*60 + "\n")


  robot_agent = RobotAgent()
  robot_agent.set_max_linear_speed(0.5)
  robot_agent.set_max_angular_speed(1.5)

  scene_graph_file = str(CV_SCENE_GRAPH_PATH)
  planner = LLMPlanner(
      agent=robot_agent,
      current_location=CV_START_LOCATION,
      scene_graph_path=scene_graph_file,
      openai_model="gpt-5.2"
  )

  rospy.loginfo("\nRobot voice agent ready. Send voice instructions!\n")

  while not rospy.is_shutdown():
      try:
          instruction = get_audio_instruction("> Instruction: ")
      except (EOFError, KeyboardInterrupt):
          break

      if is_exit_instruction(instruction):
          cleanup_transient_json_state()
          rospy.loginfo("Exit command received. Closing agent_cv.py.")
          break

      if not instruction:
          continue

      rospy.loginfo(f"[VOICE] Instruction: {instruction}")
      planner.execute_task(instruction)
      rospy.loginfo("Task completed. Ready for next instruction.\n")
