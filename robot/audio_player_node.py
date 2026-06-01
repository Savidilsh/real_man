#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS Audio Player Node for Robot Hardware
Subscribes to /robot_audio topic and plays audio files through robot speaker.
Usage: python audio_player_node.py
"""

import rospy
from std_msgs.msg import String
import subprocess
import os
from pathlib import Path
import json
import random
import re
import time

# Robot audio settings
AUDIO_DEVICE = "plughw:1,0"  # Realman robot speaker device
PREDEFINED_AUDIO_DIR = Path(__file__).resolve().parent / "sayplan" / "savindu" / "predefined_audio"


class AudioPlayerNode:
    def __init__(self):
        rospy.init_node('audio_player_node', anonymous=True)
        
        # Subscribe to /robot_audio topic
        self.sub_audio = rospy.Subscriber('/robot_audio', String, self.audio_callback, queue_size=10)
        self.pub_audio_done = rospy.Publisher('/robot_audio_done', String, queue_size=10)
        self._last_reset_attempt = 0.0
        self._reset_cooldown_sec = 5.0
        
        rospy.loginfo("=" * 60)
        rospy.loginfo("Audio Player Node Started")
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"Listening on: /robot_audio")
        rospy.loginfo(f"Audio device: {AUDIO_DEVICE}")
        rospy.loginfo(f"Predefined audio dir: {PREDEFINED_AUDIO_DIR}")
        rospy.loginfo("Ready to play robot audio...")
        rospy.loginfo("=" * 60)

    def _resolve_scenario_audio(self, scenario: str):
        if not scenario:
            return None
        if "/" in scenario or "\\" in scenario or ".." in scenario:
            rospy.logwarn(f"[AUDIO] Invalid scenario name: {scenario}")
            return None

        scenario_dir = PREDEFINED_AUDIO_DIR / scenario
        if scenario_dir.is_dir():
            candidates = sorted(scenario_dir.glob("*.wav"))
            if not candidates:
                rospy.logwarn(f"[AUDIO] No .wav files found in scenario: {scenario_dir}")
                return None
            return random.choice(candidates)

        file_path = PREDEFINED_AUDIO_DIR / f"{scenario}.wav"
        if file_path.exists():
            return file_path

        return None

    def _parse_card_index(self):
        match = re.search(r"(?:plughw|hw):(\d+)", AUDIO_DEVICE)
        if match:
            return int(match.group(1))
        return None

    def _find_usb_device_path(self, device_path: Path):
        current = device_path
        for _ in range(8):
            if (current / "idVendor").exists() and (current / "idProduct").exists():
                return current
            parent = current.parent
            if parent == current:
                break
            current = parent
        return None

    def _toggle_usb_authorized(self, usb_device_path: Path):
        authorized_path = usb_device_path / "authorized"
        if not authorized_path.exists():
            return False
        try:
            rospy.logwarn(f"[AUDIO] USB audio dismount/mount: {usb_device_path.name}")
            with open(authorized_path, "w") as f:
                f.write("0")
            time.sleep(0.5)
            with open(authorized_path, "w") as f:
                f.write("1")
            time.sleep(0.5)
            return True
        except Exception as e:
            rospy.logwarn(f"[AUDIO] USB authorization toggle failed: {e}")
            return False

    def _unbind_bind_device(self, device_path: Path):
        driver_link = device_path / "driver"
        if not driver_link.exists():
            return False
        driver_path = driver_link.resolve()
        unbind_path = driver_path / "unbind"
        bind_path = driver_path / "bind"
        if not unbind_path.exists() or not bind_path.exists():
            return False
        device_id = device_path.name
        try:
            rospy.logwarn(f"[AUDIO] Unbinding audio device {device_id} from {driver_path.name}")
            with open(unbind_path, "w") as f:
                f.write(device_id)
            time.sleep(0.5)
            rospy.logwarn(f"[AUDIO] Rebinding audio device {device_id} to {driver_path.name}")
            with open(bind_path, "w") as f:
                f.write(device_id)
            time.sleep(0.5)
            return True
        except Exception as e:
            rospy.logwarn(f"[AUDIO] Unbind/bind failed: {e}")
            return False

    def _should_reset_on_error(self, error_msg: str):
        if not error_msg:
            return False
        lowered = error_msg.lower()
        if "device or resource busy" in lowered or "resource busy" in lowered:
            return False
        return True

    def _reset_audio_device(self, error_msg: str):
        now = time.time()
        if (now - self._last_reset_attempt) < self._reset_cooldown_sec:
            rospy.logwarn("[AUDIO] Reset cooldown active; skipping audio device reset.")
            return False
        self._last_reset_attempt = now

        card_index = self._parse_card_index()
        if card_index is None:
            rospy.logwarn(f"[AUDIO] Cannot reset device for {AUDIO_DEVICE}: card index not found.")
            return False

        card_path = Path(f"/sys/class/sound/card{card_index}")
        if not card_path.exists():
            rospy.logwarn(f"[AUDIO] Audio card path not found: {card_path}")
            return False

        device_path = card_path / "device"
        if not device_path.exists():
            rospy.logwarn(f"[AUDIO] Audio device path not found: {device_path}")
            return False

        device_path = device_path.resolve()
        rospy.logwarn(f"[AUDIO] Resetting audio device due to error: {error_msg.strip()}")

        usb_device_path = self._find_usb_device_path(device_path)
        if usb_device_path and self._toggle_usb_authorized(usb_device_path):
            return True

        if self._unbind_bind_device(device_path):
            return True

        rospy.logwarn("[AUDIO] Audio device reset attempt did not succeed.")
        return False

    def _play_file(self, file_path: Path):
        rospy.loginfo(f"[AUDIO] Playing audio on device {AUDIO_DEVICE}: {file_path}")
        cmd = ['aplay', '-D', AUDIO_DEVICE, str(file_path)]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode == 0:
            rospy.loginfo("[AUDIO] Audio played successfully!")
            return True
        error_msg = result.stderr.decode('utf-8', errors='ignore')
        rospy.logerr(f"[AUDIO] Error playing audio: {error_msg}")
        if not self._should_reset_on_error(error_msg):
            rospy.logwarn("[AUDIO] Skipping audio reset for this error.")
            return False
        if self._reset_audio_device(error_msg):
            rospy.logwarn("[AUDIO] Retrying playback after audio reset...")
            retry = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            if retry.returncode == 0:
                rospy.loginfo("[AUDIO] Audio played successfully after reset!")
                return True
            retry_error = retry.stderr.decode('utf-8', errors='ignore')
            rospy.logerr(f"[AUDIO] Retry failed: {retry_error}")
        return False
        
    def audio_callback(self, msg):
        """
        Callback when audio data is received via /robot_audio topic.
        Receives base64-encoded audio, decodes it, saves to /tmp, and plays it.
        """
        audio_data = msg.data
        audio_id = None
        scenario = None
        payload = None

        try:
            payload = json.loads(audio_data)
        except Exception:
            payload = None

        if isinstance(payload, dict):
            if "audio_b64" in payload:
                audio_id = payload.get("id")
                audio_data = payload["audio_b64"]
            elif "s" in payload or "scenario" in payload or payload.get("type") == "scenario":
                scenario = payload.get("s") or payload.get("scenario") or payload.get("name")
                audio_id = payload.get("id")
            else:
                rospy.logwarn("[AUDIO] Unrecognized /robot_audio JSON payload.")
                return
        elif isinstance(payload, str):
            audio_data = payload
        elif payload is not None:
            rospy.logwarn("[AUDIO] Unsupported /robot_audio payload type.")
            return

        if scenario:
            audio_path = self._resolve_scenario_audio(scenario)
            if not audio_path:
                rospy.logerr(f"[AUDIO] Scenario not found or empty: {scenario}")
                return
            self._play_file(audio_path)
            if audio_id:
                self.pub_audio_done.publish(String(data=str(audio_id)))
            return

        rospy.loginfo(f"[AUDIO] Received audio data ({len(audio_data)} chars)")

        try:
            # Decode base64 audio data
            import base64
            audio_bytes = base64.b64decode(audio_data)
            rospy.loginfo(f"[AUDIO] Decoded {len(audio_bytes)} bytes")
            
            # Save to temporary file
            import tempfile
            import time
            temp_file = f"/tmp/robot_audio_{int(time.time())}.wav"
            
            with open(temp_file, 'wb') as f:
                f.write(audio_bytes)
            
            rospy.loginfo(f"[AUDIO] Saved to: {temp_file}")
            # Play audio with aplay (WAV)
            self._play_file(Path(temp_file))
            
            # Clean up temp file
            try:
                os.remove(temp_file)
            except:
                pass
                
        except Exception as e:
            rospy.logerr(f"[AUDIO] Error processing audio: {e}")
        finally:
            if audio_id:
                self.pub_audio_done.publish(String(data=str(audio_id)))

    def run(self):
        """Keep the node running."""
        rospy.spin()


def main():
    try:
        node = AudioPlayerNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Audio Player Node stopped.")
    except KeyboardInterrupt:
        rospy.loginfo("Audio Player Node interrupted by user.")


if __name__ == '__main__':
    main()
