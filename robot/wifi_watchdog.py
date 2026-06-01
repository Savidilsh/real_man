#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi watchdog and recovery helper.
Monitors a WiFi interface, logs status, and attempts reset when it stays down.
"""

import argparse
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path


DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "wifi_logs" / "wifi_watchdog.log"
DEFAULT_CHECK_INTERVAL_SEC = 3.0
DEFAULT_FAIL_THRESHOLD = 3
DEFAULT_RESET_COOLDOWN_SEC = 10.0


def _run_command(cmd):
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as exc:
        return 1, "", str(exc)


def _read_text(path: Path):
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _detect_wifi_interfaces():
    net_dir = Path("/sys/class/net")
    if not net_dir.exists():
        return []

    candidates = []
    for iface_path in net_dir.iterdir():
        if not iface_path.is_dir():
            continue
        if (iface_path / "wireless").exists():
            candidates.append(iface_path.name)
            continue
        if iface_path.name.startswith(("wlan", "wl")):
            candidates.append(iface_path.name)

    return sorted(set(candidates))


def _check_interface(iface: str, use_iw: bool, ping_target=None):
    iface_dir = Path("/sys/class/net") / iface
    if not iface_dir.exists():
        return False, f"interface_missing:{iface}"

    operstate = _read_text(iface_dir / "operstate")
    if operstate and operstate not in ("up", "dormant"):
        return False, f"operstate:{operstate}"

    carrier = _read_text(iface_dir / "carrier")
    if carrier and carrier != "1":
        return False, f"carrier:{carrier}"

    if use_iw and shutil.which("iw"):
        code, out, err = _run_command(["iw", "dev", iface, "link"])
        if code != 0:
            msg = err or out or "iw_error"
            return False, f"iw_link:{msg}"
        if "Not connected" in out:
            return False, "iw_not_connected"

    if ping_target:
        code, _, _ = _run_command(["ping", "-c", "1", "-W", "1", ping_target])
        if code != 0:
            return False, f"ping_fail:{ping_target}"

    return True, "ok"


def _find_usb_device_path(device_path: Path):
    current = device_path
    for _ in range(8):
        if (current / "idVendor").exists() and (current / "idProduct").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def _toggle_usb_authorized(usb_device_path: Path, logger: logging.Logger):
    authorized_path = usb_device_path / "authorized"
    if not authorized_path.exists():
        return False
    try:
        logger.warning("USB WiFi dismount/mount: %s", usb_device_path.name)
        authorized_path.write_text("0", encoding="utf-8")
        time.sleep(0.5)
        authorized_path.write_text("1", encoding="utf-8")
        time.sleep(0.5)
        return True
    except Exception as exc:
        logger.warning("USB authorization toggle failed: %s", exc)
        return False


def _unbind_bind_device(device_path: Path, logger: logging.Logger):
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
        logger.warning("Unbinding WiFi device %s from %s", device_id, driver_path.name)
        unbind_path.write_text(device_id, encoding="utf-8")
        time.sleep(0.5)
        logger.warning("Rebinding WiFi device %s to %s", device_id, driver_path.name)
        bind_path.write_text(device_id, encoding="utf-8")
        time.sleep(0.5)
        return True
    except Exception as exc:
        logger.warning("Unbind/bind failed: %s", exc)
        return False


def _snapshot_state(iface: str):
    chunks = []
    for cmd in (
        ["ip", "link", "show", iface],
        ["ip", "addr", "show", iface],
    ):
        if not shutil.which(cmd[0]):
            continue
        code, out, err = _run_command(cmd)
        chunks.append(f"$ {' '.join(cmd)} (rc={code})")
        if out:
            chunks.append(out)
        if err:
            chunks.append(err)

    if shutil.which("iw"):
        code, out, err = _run_command(["iw", "dev", iface, "link"])
        chunks.append(f"$ iw dev {iface} link (rc={code})")
        if out:
            chunks.append(out)
        if err:
            chunks.append(err)

    if shutil.which("dmesg"):
        code, out, err = _run_command(["dmesg", "--ctime", "--tail", "50"])
        chunks.append(f"$ dmesg --ctime --tail 50 (rc={code})")
        if out:
            chunks.append(out)
        if err:
            chunks.append(err)

    return "\n".join(chunks)


def _try_command(cmd, logger: logging.Logger, label: str):
    code, out, err = _run_command(cmd)
    if code != 0:
        msg = err or out or "command_failed"
        logger.warning("%s failed (rc=%d): %s", label, code, msg)
        return False
    return True


def _reset_interface(iface: str, logger: logging.Logger):
    iface_dir = Path("/sys/class/net") / iface
    device_path = iface_dir / "device"
    if not device_path.exists():
        logger.warning("WiFi device path not found for %s", iface)
        return False

    device_path = device_path.resolve()
    success = False

    if shutil.which("ip"):
        logger.warning("Soft reset via ip link: %s", iface)
        ok_down = _try_command(["ip", "link", "set", iface, "down"], logger, "ip link down")
        time.sleep(0.5)
        ok_up = _try_command(["ip", "link", "set", iface, "up"], logger, "ip link up")
        time.sleep(0.5)
        success = success or (ok_down and ok_up)

    if shutil.which("nmcli"):
        logger.warning("Soft reset via nmcli: %s", iface)
        ok_disc = _try_command(["nmcli", "dev", "disconnect", iface], logger, "nmcli disconnect")
        time.sleep(0.5)
        ok_conn = _try_command(["nmcli", "dev", "connect", iface], logger, "nmcli connect")
        time.sleep(0.5)
        success = success or (ok_disc and ok_conn)

    usb_device_path = _find_usb_device_path(device_path)
    if usb_device_path and _toggle_usb_authorized(usb_device_path, logger):
        return True

    if _unbind_bind_device(device_path, logger):
        return True

    if not success:
        logger.warning("WiFi reset completed without a confirmed recovery step.")
    return success


def _setup_logging(log_path: Path, verbose: bool):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("wifi_watchdog")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if verbose:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def main():
    parser = argparse.ArgumentParser(description="WiFi watchdog and auto-reset helper.")
    parser.add_argument("--interface", help="WiFi interface name (e.g., wlan0)")
    parser.add_argument("--log", help="Log file path")
    parser.add_argument("--interval", type=float, default=DEFAULT_CHECK_INTERVAL_SEC)
    parser.add_argument("--fail-threshold", type=int, default=DEFAULT_FAIL_THRESHOLD)
    parser.add_argument("--reset-cooldown", type=float, default=DEFAULT_RESET_COOLDOWN_SEC)
    parser.add_argument("--ping-target", help="Optional ping target to verify connectivity")
    parser.add_argument("--no-iw-check", action="store_true", help="Disable iw link check")
    parser.add_argument("--verbose", action="store_true", help="Also log to stdout")
    args = parser.parse_args()

    iface = args.interface or os.getenv("WIFI_INTERFACE")
    if not iface:
        candidates = _detect_wifi_interfaces()
        iface = candidates[0] if candidates else None

    if not iface:
        raise SystemExit("WiFi interface not found. Set --interface or WIFI_INTERFACE.")

    log_path = Path(args.log) if args.log else Path(os.getenv("WIFI_LOG_PATH", str(DEFAULT_LOG_PATH)))
    logger = _setup_logging(log_path, args.verbose)

    logger.info("WiFi watchdog started for interface: %s", iface)
    logger.info("Log file: %s", log_path)

    fail_count = 0
    last_reset = 0.0

    while True:
        ok, reason = _check_interface(iface, not args.no_iw_check, args.ping_target)
        if ok:
            if fail_count:
                logger.info("WiFi recovered: %s", reason)
            fail_count = 0
        else:
            fail_count += 1
            logger.warning("WiFi check failed (%s) [%d/%d]", reason, fail_count, args.fail_threshold)
            now = time.time()
            if fail_count >= args.fail_threshold and (now - last_reset) >= args.reset_cooldown:
                logger.warning("Resetting WiFi interface after failures: %s", iface)
                snapshot = _snapshot_state(iface)
                if snapshot:
                    logger.info("WiFi snapshot:\n%s", snapshot)
                _reset_interface(iface, logger)
                last_reset = now
                fail_count = 0

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
