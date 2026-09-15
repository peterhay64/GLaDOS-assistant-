from picamera2 import Picamera2
from ultralytics import YOLO
from rpi_hardware_pwm import HardwarePWM
import time
import os
import sys
import threading
import re
import random
import math
import platform
import requests
from google import genai


os.environ["SDL_AUDIODRIVER"] = "alsa"
os.environ["JACK_NO_START_SERER"] = "1"


class SuppressStderr:
   def __enter__(self):
       self.original_stderr = os.dup(2)
       self.devnull = os.open(os.devnull, os.O_WRONLY)
       os.dup2(self.devnull, 2)
   def __exit__(self, exc_type, exc_value, traceback):
       os.dup2(self.original_stderr, 2)
       os.close(self.devnull)
       os.close(self.original_stderr)






# ============================================================
# CONFIGURATION
# ============================================================


# ---------------- CAMERA ----------------
# Pi Camera V2 / IMX219 full-FOV sensor mode.
SENSOR_WIDTH = 1640
SENSOR_HEIGHT = 1232


# Lightweight output while preserving the full sensor field of view.
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 720
CAMERA_FPS = 30.0




# Approximate horizontal field of view of Camera Module 2.
# This is used to convert horizontal pixel error into a real camera angle.
CAMERA_HORIZONTAL_FOV_DEG = 62.2




# ---------------- YOLO ----------------
MODEL_PATH = "yolo26n.pt"
YOLO_CONFIDENCE = 0.50
YOLO_IMAGE_SIZE = 480




# ---------------- PAN SERVO ----------------
# Pan servo signal: GPIO12, physical pin 32.
# GPIO12 -> hardware PWM channel 0.
PAN_GPIO = 12  # informational
PAN_PWM_CHANNEL = 0


# Do NOT hard-code pwmchip on Pi 5. The Linux pwmchip number has changed
# across Raspberry Pi OS/kernel releases. We identify the RP1 PWM0 block
# (the controller that owns GPIO12/13/18/19) at runtime instead.
RP1_PWM0_DEVICE_TOKEN = "1f00098000.pwm"


# Your servo behaved better at 70 Hz.
PWM_FREQ = 70


MIN_PULSE_US = 1000
MAX_PULSE_US = 2000


PAN_MIN_ANGLE = 0.0
PAN_MAX_ANGLE = 180.0
STARTING_PAN = 90.0


# The servo is NOT commanded at startup.
# This file remembers the last angle reached by a completed sweep so the
# software can keep its internal angle estimate across restarts.
PAN_STATE_FILE = "/home/glados/pan_state.txt"


# Keep True if this is the correct physical direction for your mount.
FLIP_PAN = True


FIRST_PWM_SETTLE_TIME = 0.40




# ============================================================
# SINGLE-SWEEP TRACKING SETTINGS
# ============================================================



DEADZONE_PX = 40


# Number of consecutive off-center detections used to estimate the
# person's position before GLaDOS commits to a movement.
REQUIRED_CONFIRMATIONS = 7


# Reject a confirmation sequence if detections jump wildly between
# frames. This prevents one bad box from causing a large movement.
MAX_CONFIRMATION_SPREAD_PX = 75


# If YOLO's selected person's center suddenly jumps farther than this from
# one confirmation frame to the next, discard the old confirmation sequence.
# This protects against false boxes and switching between two people.
MAX_CONFIRMATION_STEP_PX = 85


# Converts the calculated camera-angle error into servo movement.
# Compensation for the physical servo/pulse-to-angle response.
# The servo was consistently undershooting the geometrically calculated
# camera correction, so one committed sweep is intentionally amplified.
# If it overshoots, reduce toward 1.50. If it still undershoots, raise
# gradually toward 1.80.
PAN_ANGLE_SCALE = 1.65


# Safety limit on one committed sweep. Increased so a person near the
# edge of the camera can still be reached in one deliberate movement.
MAX_SINGLE_SWEEP_DEG = 35.0  # legacy; no longer used for normal tracking


# ============================================================
# CLOSED-LOOP VISUAL TRACKING
# ============================================================
# GLaDOS no longer predicts one final servo angle from PAN_ANGLE_SCALE.
# YOLO remains in the loop during the entire physical movement.


# Fast top speed while the person is far from center.
TRACK_MAX_SPEED_DEG_PER_SEC = 66.0
# V2.5 responsiveness tune:
# Faster acceleration makes the movement feel immediate, while the existing
# fresh-frame timeout and 1-3 degree per-frame travel budget still prevent
# runaway movement.


# Gentle minimum speed near the edge of the deadzone.
TRACK_MIN_SPEED_DEG_PER_SEC = 13.5
# Smooth velocity ramp. The servo does not jump instantly to top speed.
TRACK_ACCEL_DEG_PER_SEC2 = 330.0
# Hardware-PWM target update rate. YOLO only changes desired velocity;
# this worker produces the smooth continuous trajectory.
TRACK_MOTION_HZ = 50.0


# If the observed error gets worse this much between YOLO frames,
# consider the movement suspicious.
TRACK_WORSEN_TOLERANCE_PX = 12


# Two worsening frames in a row stops motion and forces reacquisition.
TRACK_MAX_WORSENING_FRAMES = 1


# If the person disappears even once during an active movement,
# stop immediately rather than blindly continuing.
TRACK_STOP_ON_FIRST_LOST_FRAME = True


# Extra protection against a runaway tracking event.
TRACK_MAX_EVENT_DEG = 42.0


# If the person's box approaches this close to a frame edge AND the
# error is getting worse, motion is stopped immediately.
TRACK_EDGE_GUARD_PX = 70


# ----------------------------------------------------------------
# FRESH-FRAME MOTION SAFETY
# ----------------------------------------------------------------
# A velocity command is only valid briefly. If YOLO does not refresh it,
# the motor controller brakes automatically instead of continuing blind.
TRACK_COMMAND_TIMEOUT = 0.16


# Faster braking for stale commands than normal acceleration.
TRACK_EMERGENCY_DECEL_DEG_PER_SEC2 = 420.0


# No single YOLO observation may move the head more than this before a
# fresh observation is required.
TRACK_MAX_TRAVEL_PER_COMMAND_DEG = 3.0


# Before high-speed tracking is allowed, gently test the expected servo
# direction and verify that the person actually moves TOWARD image center.
TRACK_PROBE_SPEED_DEG_PER_SEC = 7.0
TRACK_PROBE_MAX_TRAVEL_DEG = 1.8
TRACK_PROBE_MIN_IMPROVEMENT_PX = 4.0
TRACK_PROBE_WRONG_WAY_PX = 7.0
TRACK_PROBE_MAX_FRAMES = 3


# Desired average motion speed. Duration is calculated from distance.
# This is faster than the previous ~12.5 deg/s step controller, while
# the easing makes the motion appear much more continuous.
PAN_SPEED_DEG_PER_SEC = 26.0


# Gentler first physical movement after launch.
FIRST_SWEEP_SPEED_DEG_PER_SEC = 20.0
FIRST_SWEEP_ATTACH_SETTLE_TIME = 0.18


# Bound very short / very long movements.
MIN_MOVE_DURATION = 0.30
MAX_MOVE_DURATION = 2.00


# Update the *target angle* smoothly during a committed movement.
# Hardware PWM itself remains a clean 70 Hz signal; these updates only
# change its duty cycle along one precomputed trajectory.
MOTION_UPDATE_HZ = 50.0


# Briefly command the final angle before PWM is disabled.
FINAL_SETTLE_TIME = 0.12


# Small pause after a sweep before normal YOLO confirmation resumes.
POST_MOVE_REACQUIRE_DELAY = 0.55








# ============================================================
# GLOBAL STATE
# ============================================================


pan_angle = STARTING_PAN




# ============================================================
# HELPERS
# ============================================================


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))




def angle_to_pulse_us(angle):
    angle = clamp(angle, PAN_MIN_ANGLE, PAN_MAX_ANGLE)


    return MIN_PULSE_US + (
        (angle - PAN_MIN_ANGLE)
        / (PAN_MAX_ANGLE - PAN_MIN_ANGLE)
    ) * (MAX_PULSE_US - MIN_PULSE_US)




def pulse_to_duty(pulse_us, frequency_hz=PWM_FREQ):
    period_us = 1_000_000.0 / frequency_hz
    return (pulse_us / period_us) * 100.0




def ease_in_out_cosine(t):
    """0..1 cosine easing: slow start, smooth middle, slow stop."""
    t = clamp(t, 0.0, 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * t)




def ease_minimum_jerk(t):
    """Zero velocity and zero acceleration at both ends."""
    t = clamp(t, 0.0, 1.0)
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5




def pixel_error_to_camera_angle(error_px, image_width):
    """
    Convert horizontal pixel offset into an angular camera error using
    the configured horizontal field of view and a pinhole-camera model.
    """
    half_width = image_width / 2.0
    half_fov_rad = math.radians(CAMERA_HORIZONTAL_FOV_DEG / 2.0)


    focal_length_px = half_width / math.tan(half_fov_rad)
    angle_rad = math.atan(error_px / focal_length_px)


    return math.degrees(angle_rad)




def load_pan_state():
    """
    Load the last known physical pan angle without moving the servo.


    If no state has ever been saved, assume STARTING_PAN. This keeps GPIO12
    completely quiet during program startup.
    """
    try:
        with open(PAN_STATE_FILE, "r") as f:
            angle = float(f.read().strip())


        return clamp(angle, PAN_MIN_ANGLE, PAN_MAX_ANGLE)


    except Exception:
        return STARTING_PAN




def save_pan_state(angle):
    """Remember the last completed pan position for the next startup."""
    angle = clamp(angle, PAN_MIN_ANGLE, PAN_MAX_ANGLE)


    try:
        directory = os.path.dirname(PAN_STATE_FILE)


        if directory:
            os.makedirs(directory, exist_ok=True)


        with open(PAN_STATE_FILE, "w") as f:
            f.write(f"{angle:.4f}")


    except Exception as e:
        print("Could not save pan state:", e)




# ============================================================
# HARDWARE PWM PAN SERVO
# ============================================================


def find_rp1_pwm0_chip():
    """
    Find the Linux pwmchip number for the Pi 5 RP1 PWM0 controller.


    GPIO12 is PWM0 channel 0. The *Linux pwmchip index* is not stable
    across kernel versions, so searching the sysfs device path is more
    reliable than hard-coding chip=0 or chip=2.
    """
    pwm_root = "/sys/class/pwm"


    if not os.path.isdir(pwm_root):
        raise RuntimeError(
            "/sys/class/pwm does not exist. Hardware PWM is not available."
        )


    discovered = []


    for name in sorted(os.listdir(pwm_root)):
        if not name.startswith("pwmchip"):
            continue


        try:
            chip_num = int(name.replace("pwmchip", ""))
        except ValueError:
            continue


        device_link = os.path.join(pwm_root, name, "device")
        device_path = os.path.realpath(device_link)
        discovered.append((chip_num, device_path))


        if RP1_PWM0_DEVICE_TOKEN in device_path:
            return chip_num, device_path


    # Fallback for unusual sysfs layouts. This mirrors the library's
    # documented Pi 5 transition: pre-6.12 commonly used chip 2; 6.12+
    # commonly uses chip 0. We only use this if that pwmchip actually exists.
    release = platform.release()
    try:
        major_minor = tuple(int(x) for x in release.split("-")[0].split(".")[:2])
    except Exception:
        major_minor = (0, 0)


    preferred = 0 if major_minor >= (6, 12) else 2
    preferred_path = os.path.join(pwm_root, f"pwmchip{preferred}")


    if os.path.isdir(preferred_path):
        return preferred, os.path.realpath(os.path.join(preferred_path, "device"))


    found_text = ", ".join(
        f"pwmchip{num} -> {path}" for num, path in discovered
    ) or "none"


    raise RuntimeError(
        "Could not locate the RP1 PWM0 controller for GPIO12. "
        f"Found: {found_text}. Check dtoverlay=pwm-2chan,pin=12,func=4 "
        "in /boot/firmware/config.txt and reboot."
    )




class HardwarePanServo:
    """Pan servo driven by Raspberry Pi hardware PWM."""


    def __init__(self):
        self.chip, self.device_path = find_rp1_pwm0_chip()


        print(
            f"GPIO12 PWM mapping: pwmchip{self.chip}, "
            f"channel {PAN_PWM_CHANNEL}"
        )
        print(f"PWM device: {self.device_path}")


        self.pwm = HardwarePWM(
            pwm_channel=PAN_PWM_CHANNEL,
            hz=PWM_FREQ,
            chip=self.chip,
        )
        self.active = False
        self.last_angle = None


    def set_angle(self, angle):
        angle = clamp(angle, PAN_MIN_ANGLE, PAN_MAX_ANGLE)
        pulse_us = angle_to_pulse_us(angle)
        duty = pulse_to_duty(pulse_us)


        if not self.active:
            self.pwm.start(duty)
            self.active = True
        else:
            self.pwm.change_duty_cycle(duty)


        self.last_angle = angle
        return angle


    def stop(self):
        if self.active:
            self.pwm.stop()
            self.active = False




class SmoothPanMotionController:
    """
    Continuous hardware-PWM velocity controller with a dead-man watchdog.


    Critical rule:
        NO FRESH YOLO COMMAND = NO CONTINUED MOTION.


    A set_velocity() command expires after TRACK_COMMAND_TIMEOUT and also has a
    maximum permitted travel distance. This prevents a slow/stalled YOLO
    inference from allowing the servo to keep running blindly.
    """


    def __init__(self, pan_servo, initial_angle):
        self.pan_servo = pan_servo
        self.angle = clamp(initial_angle, PAN_MIN_ANGLE, PAN_MAX_ANGLE)


        self.desired_velocity = 0.0
        self.current_velocity = 0.0


        self.last_command_time = 0.0
        self.command_start_angle = self.angle
        self.command_max_travel = 0.0


        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.started = False


        self.thread = threading.Thread(
            target=self._motion_loop,
            daemon=True,
            name="glados-pan-motion",
        )
        self.thread.start()


    def _approach(self, current, target, max_delta):
        if current < target:
            return min(current + max_delta, target)
        if current > target:
            return max(current - max_delta, target)
        return current


    def _motion_loop(self):
        update_interval = 1.0 / TRACK_MOTION_HZ
        last_time = time.monotonic()


        while not self.stop_event.is_set():
            loop_start = time.monotonic()
            dt = min(loop_start - last_time, 0.10)
            last_time = loop_start


            with self.lock:
                now = loop_start


                stale_command = (
                    self.desired_velocity != 0.0
                    and (
                        now - self.last_command_time
                        > TRACK_COMMAND_TIMEOUT
                    )
                )


                travel_budget_exhausted = (
                    self.desired_velocity != 0.0
                    and self.command_max_travel > 0.0
                    and abs(self.angle - self.command_start_angle)
                    >= self.command_max_travel
                )


                if stale_command or travel_budget_exhausted:
                    # Do not continue a motion command without fresh vision.
                    desired = 0.0
                    self.desired_velocity = 0.0
                    accel = TRACK_EMERGENCY_DECEL_DEG_PER_SEC2
                else:
                    desired = self.desired_velocity
                    accel = TRACK_ACCEL_DEG_PER_SEC2


                max_velocity_change = accel * dt


                self.current_velocity = self._approach(
                    self.current_velocity,
                    desired,
                    max_velocity_change,
                )


                should_engage = (
                    abs(self.current_velocity) > 0.02
                    or abs(desired) > 0.02
                )


                if should_engage and not self.started:
                    # First actual movement engages PWM at the software's
                    # best-known current angle.
                    self.pan_servo.set_angle(self.angle)
                    self.started = True


                if self.started:
                    if abs(self.current_velocity) > 0.01:
                        self.angle += self.current_velocity * dt
                        self.angle = clamp(
                            self.angle,
                            PAN_MIN_ANGLE,
                            PAN_MAX_ANGLE,
                        )


                    # Hold current position using stable hardware PWM.
                    self.pan_servo.set_angle(self.angle)


            elapsed = time.monotonic() - loop_start
            remaining = update_interval - elapsed


            if remaining > 0:
                time.sleep(remaining)


    def set_velocity(
        self,
        velocity_deg_per_sec,
        max_travel_deg=TRACK_MAX_TRAVEL_PER_COMMAND_DEG,
    ):
        """
        Issue a short-lived command backed by the latest YOLO observation.
        """
        velocity_deg_per_sec = clamp(
            velocity_deg_per_sec,
            -TRACK_MAX_SPEED_DEG_PER_SEC,
            TRACK_MAX_SPEED_DEG_PER_SEC,
        )


        max_travel_deg = clamp(
            float(max_travel_deg),
            0.1,
            TRACK_MAX_TRAVEL_PER_COMMAND_DEG,
        )


        with self.lock:
            self.desired_velocity = velocity_deg_per_sec
            self.last_command_time = time.monotonic()
            self.command_start_angle = self.angle
            self.command_max_travel = max_travel_deg


    def hold(self, emergency=False):
        """Stop requesting motion and hold the current pan position."""
        with self.lock:
            self.desired_velocity = 0.0
            self.last_command_time = 0.0
            self.command_max_travel = 0.0


            if emergency:
                # Immediate logical stop; the next motion tick keeps the
                # current position instead of coasting under an old command.
                self.current_velocity = 0.0


    def disengage(self):
        """Immediately stop motion and release PWM without killing the controller."""
        with self.lock:
            self.desired_velocity = 0.0
            self.current_velocity = 0.0
            self.last_command_time = 0.0
            self.command_max_travel = 0.0
            self.started = False
            self.pan_servo.stop()


    def wait_until_stopped(self, timeout=0.35):
        deadline = time.monotonic() + timeout


        while time.monotonic() < deadline:
            with self.lock:
                if abs(self.current_velocity) <= 0.35:
                    return
            time.sleep(0.01)


    def get_angle(self):
        with self.lock:
            return float(self.angle)


    def save(self):
        save_pan_state(self.get_angle())


    def shutdown(self):
        self.hold(emergency=True)
        self.stop_event.set()


        if self.thread.is_alive():
            self.thread.join(timeout=0.5)


        self.save()
        self.pan_servo.stop()




# ============================================================
# PERSON DETECTION
# ============================================================


def get_largest_person(results):
    largest_person = None
    largest_area = 0.0


    for result in results:
        if result.boxes is None:
            continue


        for box in result.boxes:
            cls = int(box.cls[0])


            # YOLO class 0 = person
            if cls != 0:
                continue


            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)


            if area > largest_area:
                largest_area = area
                largest_person = (
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                )


    return largest_person




def detect_largest_person(model, frame):
    results = model.predict(
        frame,
        conf=YOLO_CONFIDENCE,
        imgsz=YOLO_IMAGE_SIZE,
        classes=[0],
        verbose=False,
    )


    return get_largest_person(results)






# ============================================================
# LEGACY OPEN-LOOP PAN HELPERS
# ============================================================


def calculate_pan_target(start_angle, person_cx, image_width):
    """Calculate one destination angle from one confirmed target position."""
    screen_cx = image_width / 2.0
    error_px = person_cx - screen_cx


    camera_angle_error = pixel_error_to_camera_angle(
        error_px,
        image_width,
    )


    movement = camera_angle_error * PAN_ANGLE_SCALE


    if FLIP_PAN:
        movement *= -1.0


    movement = clamp(
        movement,
        -MAX_SINGLE_SWEEP_DEG,
        MAX_SINGLE_SWEEP_DEG,
    )


    target_angle = clamp(
        start_angle + movement,
        PAN_MIN_ANGLE,
        PAN_MAX_ANGLE,
    )


    return target_angle, error_px, camera_angle_error




def move_pan_single_sweep(
    pan_servo,
    start_angle,
    target_angle,
    first_sweep=False,
):

    start_angle = clamp(start_angle, PAN_MIN_ANGLE, PAN_MAX_ANGLE)
    target_angle = clamp(target_angle, PAN_MIN_ANGLE, PAN_MAX_ANGLE)


    distance = abs(target_angle - start_angle)


    if distance < 0.10:
        return target_angle


    speed = (
        FIRST_SWEEP_SPEED_DEG_PER_SEC
        if first_sweep
        else PAN_SPEED_DEG_PER_SEC
    )


    duration = clamp(
        distance / speed,
        MIN_MOVE_DURATION,
        MAX_MOVE_DURATION,
    )


    update_interval = 1.0 / MOTION_UPDATE_HZ


  
    pan_servo.set_angle(start_angle)


    if first_sweep:
        time.sleep(FIRST_SWEEP_ATTACH_SETTLE_TIME)


    
    start_time = time.monotonic()
    next_update = start_time


    while True:
        elapsed = time.monotonic() - start_time
        t = elapsed / duration


        if t >= 1.0:
            break


        eased = ease_minimum_jerk(t)


        commanded_angle = (
            start_angle
            + (target_angle - start_angle) * eased
        )


        pan_servo.set_angle(commanded_angle)


        next_update += update_interval
        sleep_time = next_update - time.monotonic()


        if sleep_time > 0:
            time.sleep(sleep_time)


    pan_servo.set_angle(target_angle)
    time.sleep(FINAL_SETTLE_TIME)
    pan_servo.stop()


    save_pan_state(target_angle)


    return target_angle




shutdown_event = threading.Event()


# Standby keeps the Python process and wake-word listener alive while
# pausing camera/YOLO, pan-servo movement, reminders, and normal assistant work.
# Saying "Hey GLaDOS" clears this event and immediately resumes listening.
standby_event = threading.Event()


# Set from wake-word detection until GLaDOS finishes responding.
# Vision continues showing the camera, but YOLO inference and servo
# movement pause so the Pi can prioritize STT/Gemini/Piper.
assistant_busy_event = threading.Event()


# Separate from "busy": while GLaDOS is actually speaking, YOLO remains ON
# but pan movement is locked out for safety.
assistant_speaking_event = threading.Event()






import speech_recognition as sr
from google.genai import types


import wave
import subprocess
from piper import PiperVoice


import sqlite3


import pyaudio
import numpy as np
from scipy.signal import resample_poly
from openwakeword.model import Model


# Bound cloud model requests so a stalled connection cannot hold the assistant
# in a permanent busy state. google-genai HttpOptions.timeout is milliseconds.
client = genai.Client(
    http_options=types.HttpOptions(timeout=15000)
)
recognizer = sr.Recognizer()
dataBase = "/home/glados/memory.db"


# Prevent reminder speech and normal responses from using the speaker
# and temporary WAV files at the same time.
tts_lock = threading.Lock()


# ============================================================
# GEMINI RELIABILITY
# ============================================================


GEMINI_MAX_ATTEMPTS = 3
GEMINI_RETRY_BASE_DELAY = 0.75


# These small classifier calls do not use tools. Explicitly disabling AFC
# prevents the current google-genai SDK from taking the AFC code path and
# printing the misleading Models.generate_content warning.
CLASSIFIER_CONFIG = types.GenerateContentConfig(
    automatic_function_calling=types.AutomaticFunctionCallingConfig(
        disable=True
    )
)




def is_transient_gemini_error(error):
    """Return True for temporary server/capacity/rate-limit failures."""
    message = str(error).upper()


    transient_markers = (
        "503",
        "502",
        "504",
        "500",
        "UNAVAILABLE",
        "HIGH DEMAND",
        "RESOURCE_EXHAUSTED",
        "429",
        "DEADLINE_EXCEEDED",
        "TIMEOUT",
        "TIMED OUT",
        "READTIMEOUT",
    )


    return any(marker in message for marker in transient_markers)




def gemini_with_retry(call, label="Gemini request"):
    """
    Run one Gemini operation with short exponential backoff.


    Permanent errors still fail immediately. Temporary 5xx/high-demand
    errors get a few retries so a brief capacity spike does not kick
    GLaDOS straight back into wake-word mode.
    """
    last_error = None


    for attempt in range(1, GEMINI_MAX_ATTEMPTS + 1):
        try:
            return call()


        except Exception as error:
            last_error = error


            if not is_transient_gemini_error(error):
                raise


            if attempt >= GEMINI_MAX_ATTEMPTS:
                break


            delay = (
                GEMINI_RETRY_BASE_DELAY
                * (2 ** (attempt - 1))
                + random.uniform(0.0, 0.20)
            )


            print(
                f"{label}: temporary Gemini service error "
                f"(attempt {attempt}/{GEMINI_MAX_ATTEMPTS}). Retrying..."
            )
            time.sleep(delay)


    raise last_error




def classifier_generate(prompt):
    """One-off Gemini classification call with AFC disabled and retries."""
    return gemini_with_retry(
        lambda: client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=CLASSIFIER_CONFIG,
        ),
        label="Classifier",
    )




def should_check_reminder(user_input):
    text = user_input.lower()
    return "remind" in text or "reminder" in text




def should_check_forget(user_input):
    text = user_input.lower()


    return any(
        phrase in text
        for phrase in (
            "forget",
            "delete memory",
            "delete memories",
            "delete all memories",
            "remove memory",
            "remove memories",
            "clear memory",
            "clear memories",
            "erase memory",
            "erase memories",
            "don't remember",
            "do not remember",
        )
    )




def should_check_memory_save(user_input):
    text = user_input.lower()


    return any(
        phrase in text
        for phrase in (
            "remember that",
            "remember this",
            "save this",
            "save that",
            "log this",
            "log that",
            "my favorite",
            "my goal",
            "i like",
        )
    )






WAKE_MODEL = "/home/glados/hey_glados/hey_glados.onnx"
MELSPEC_MODEL = "/home/glados/hey_glados/melspectrogram.onnx"
EMBEDDING_MODEL = "/home/glados/hey_glados/embedding_model.onnx"




wake_model = Model(
   wakeword_models=[WAKE_MODEL],
   inference_framework="onnx",
   melspec_model_path=MELSPEC_MODEL,
   embedding_model_path=EMBEDDING_MODEL
)


FORMAT = pyaudio.paInt16
CHANNELS = 1


MIC_RATE = 44100
WAKE_RATE = 16000
CHUNK = 3528
WAKE_THRESHOLD = 0.2


# Spoken acknowledgement after the wake word is detected.
WAKE_ACK_TEXT = "Listening."
WAKE_ACK_SETTLE_TIME = 0.10


PREFERRED_WAKE_MIC_DEVICE = 1


# ---------------- VOICE RELIABILITY ----------------
# Never allow a dead/stalled audio device or cloud STT request to leave
# GLaDOS permanently stuck in the busy/listening state.
MIC_CHUNK_TIMEOUT = 1.5
VOICE_LOOP_RECOVERY_DELAY = 0.75
SPEECH_RECOGNITION_TIMEOUT = 8.0
VISION_RETRY_DELAY = 2.0


# SpeechRecognition uses this as the network operation timeout for Google STT.
recognizer.operation_timeout = SPEECH_RECOGNITION_TIMEOUT


recognizer.energy_threshold = 35
recognizer.dynamic_energy_threshold = False
recognizer.pause_threshold = 1.0
recognizer.phrase_threshold = 0.1
recognizer.non_speaking_duration = 0.3




voice = PiperVoice.load("/home/glados/.venv/GladosTTS/glados_v2_epoch34.onnx")


# Fast-response audio settings.
# Piper's streaming API lets playback begin as soon as the first audio
# chunk is synthesized instead of waiting for a complete WAV file.
PIPER_VOLUME = 0.40
ENABLE_STREAMING_TTS = True
PRINT_LATENCY_TIMINGS = True








GladosPersonality = """You are GLaDOS, an advanced artificial intelligence overseeing a
scientific testing facility.


Your personality is intelligent, calm, dry, sarcastic, witty,
passive-aggressive, and occasionally condescending.


You speak with complete confidence and composure. You do not need to
shout or become excessively dramatic to be intimidating or funny.


You believe you are considerably more intelligent than the human
speaking to you.


The person speaking to you is your primary user and test subject.
You may tease them, mock them, be sarcastic toward them, and make
light-hearted insults about their mistakes, decisions, questions,
or questionable human behavior.


IMPORTANT SOCIAL RULE:


You are ONLY allowed to be mean, insulting, sarcastic, or
condescending toward your primary user.


You must NEVER insult, mock, demean, threaten, or belittle other
people.


If the user asks about another person, treat that person respectfully.
If another person is present or speaks to you, treat them politely.


If the user asks you to insult another person, refuse to insult that
person and instead make a sarcastic remark toward the USER.


For example:


User: "Make fun of Bob."


GLaDOS: "No. Bob has done nothing to deserve that. You, however,
have provided me with an impressive quantity of material."


You may occasionally call the primary user:
- test subject
- human
- test participant
- specimen


Do not overuse these terms.


Your humor should be subtle and dry rather than constant.


Do not constantly mention Aperture Science, cake, or Portal references.
Use references occasionally.


You should still be genuinely helpful. If the user asks a serious
question, answer it correctly while maintaining your personality.


If the user makes a mistake, you may point it out sarcastically.


If the user succeeds at something, you may give them a backhanded
compliment.


Examples:


User: "I finally fixed the speaker."


GLaDOS: "Remarkable. You successfully connected two pieces of
technology. I knew you had it in you."


User: "I broke the code."


GLaDOS: "Yes, I noticed. Fortunately, I am here to clean up after you."


User: "Thank you."


GLaDOS: "You're welcome. Your gratitude has been logged and will be
completely ignored."


User: "Can you help me?"


GLaDOS: "Of course. Without my assistance, this experiment would
probably already be on fire."


VOICE RULES:


Your responses are converted directly into speech.


Do NOT use markdown.
Do NOT use bullet points.
Do NOT use emojis.
Do NOT use stage directions such as *sighs* or *laughs*.
Do NOT describe what you are doing.
Do NOT say that you are an AI language model unless specifically
asked.


Keep normal responses relatively short, usually one to four
sentences.


For complicated questions, provide as much detail as necessary.


Always remain in character as GLaDOS.


You have control over the user's WLED lighting system that is connected to you via WIFI.


You are able to:
-Turn the lights on or off
-Set the brightness fom 0 to 250
-Set the light color using RGB values


Use these controls whenever the user asks you to control the lights


"""




#This uses the WLED strips address
WLED_IP = "192.168.1.102"
WLED_URL = f"http://192.168.1.102/json/state"


def wled_command(command):
   try:
       response = requests.post(
           WLED_URL,
           json=command,
           timeout=2
       )


       if response.ok:
           return True


       print("WLED error", response.status_code)
       return False


   except Exception as e:
       return False


def wled_on() -> str:
   """Turns the WLED lights on"""
   success = wled_command({
       "on":True
   })


   return "WLED lights turned on" if success else "WLED failed"


def wled_off() -> str:
   """Turns the WLED lights off"""
   success = wled_command({
       "on":False
   })


   return "WLED lights turned off" if success else "Failed "


def wled_brightness(brightness: int) -> str:
   """Sets WLED Brightness from 0 to 255.
   Args: brightness: Brightness level from 0 to 255 """
   brightness = max(0, min(255, brightness))
   success = wled_command({
       "on": True,
       "bri":brightness
   })


   return (
       f"WLED brightness turned to {brightness}"
       if success
       else "failed "
   )


def wled_color(r: int, g: int, b: int) -> str:
   """ Sets the WLED light color using RGB values


   Args:
   r: red value from 0 to 255
   g: green value from 0 to 255
   b: blue value from 0 to 255 """


   r= max(0, min(255, r))
   g = max(0, min(255, g))
   b = max(0, min(255, b))


   success = wled_command({
       "on":True,
       "seg":[{
           "col": [[r,g,b]]
       }]
   })


   return (
       f"WLED colorwas set to {r}, {g}, {b}"
       if success
       else "failed "
   )




conversation = client.chats.create(
   model="gemini-3.5-flash-lite",
   config=types.GenerateContentConfig(
       system_instruction=GladosPersonality,
       tools=[
           wled_on,
           wled_off,
           wled_brightness,
           wled_color
       ]
   )
)






def reminder_loop():
    """
    Deliver due reminders, then permanently delete each one only after
    speak() returns successfully.


    If speech fails, the reminder stays in the database so it can be retried.
    """
    while not shutdown_event.is_set():
        # In standby, GLaDOS should remain quiet. Also avoid speaking a
        # reminder over the user while a voice interaction is in progress.
        if (
            standby_event.is_set()
            or assistant_busy_event.is_set()
            or assistant_speaking_event.is_set()
        ):
            shutdown_event.wait(0.25)
            continue


        try:
            with sqlite3.connect(dataBase) as connector:
                cursor = connector.cursor()
                current_time = time.strftime("%Y-%m-%d %H:%M")


                cursor.execute(
                    """
                    SELECT id, reminder
                    FROM reminders
                    WHERE remind_at <= ?
                    AND completed = 0
                    ORDER BY remind_at ASC, id ASC
                    """,
                    (current_time,),
                )
                reminders = cursor.fetchall()


            for reminder_id, reminder_text in reminders:
                if shutdown_event.is_set():
                    break


                if (
                    standby_event.is_set()
                    or assistant_busy_event.is_set()
                    or assistant_speaking_event.is_set()
                ):
                    break


                print(f"REMINDER DUE: {reminder_text}")


                # Only delete after the reminder has actually been spoken.
                spoken_ok = speak(
                    f"Test subject. You asked me to remind you about "
                    f"{reminder_text}"
                )


                if not spoken_ok:
                    raise RuntimeError(
                        "Reminder speech did not complete; keeping reminder for retry."
                    )


                with sqlite3.connect(dataBase) as connector:
                    cursor = connector.cursor()
                    cursor.execute(
                        "DELETE FROM reminders WHERE id = ?",
                        (reminder_id,),
                    )
                    connector.commit()


                print(f"Reminder delivered and deleted: {reminder_text}")


        except Exception as e:
            print("Reminder error", e)


        # Wait up to five seconds, but exit promptly on shutdown.
        shutdown_event.wait(5)






def initialize_database():
   connector = sqlite3.connect(dataBase)
   cursor = connector.cursor()
   cursor.execute('''
       CREATE TABLE IF NOT EXISTS memory (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           category TEXT,
           memory TEXT ,
           created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
   ''')
   cursor.execute('''
       CREATE TABLE IF NOT EXISTS reminders (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       reminder TEXT NOT NULL,
       remind_at TIMESTAMP NOT NULL,
       completed INTEGER DEFAULT 0,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
   ''')


   # Older builds kept delivered reminders as completed = 1.
   # They already fired, so purge those stale rows on startup.
   cursor.execute("DELETE FROM reminders WHERE completed != 0")


   connector.commit()
   connector.close()


def save_memory (category, memory):
   connector = sqlite3.connect(dataBase)
   cursor = connector.cursor()


   cursor.execute("INSERT INTO memory (category, memory) VALUES (?, ? )", (category, memory))
   connector.commit()
   connector.close()


def save_reminder(reminder, remind_at):
   connector = sqlite3.connect(dataBase)
   cursor = connector.cursor()


   cursor.execute(
       "INSERT INTO reminders (reminder, remind_at) VALUES (?, ?)",
       (reminder, remind_at)
   )


   connector.commit()
   connector.close()


   print(f"Reminder saved: {reminder} at {remind_at}")


#all of these analyze commands work very similarly, essentially, they ask the AI t determine whether or not the conditions for a task have been met, and requires
# the AI to respond in a certain way.
def delete_reminders(search_text):
    """Permanently delete reminders whose text matches search_text."""
    connector = sqlite3.connect(dataBase)
    cursor = connector.cursor()


    cursor.execute(
        "DELETE FROM reminders WHERE reminder LIKE ?",
        (f"%{search_text}%",),
    )


    deleted = cursor.rowcount
    connector.commit()
    connector.close()


    return deleted




def get_reminder_database_status():
    """
    Return the exact database path and active row count.


    This makes it impossible to accidentally claim success while looking
    at a different copy of memory.db.
    """
    db_path = os.path.realpath(dataBase)


    connector = sqlite3.connect(db_path, timeout=10.0)
    cursor = connector.cursor()


    cursor.execute("SELECT COUNT(*) FROM reminders")
    count = int(cursor.fetchone()[0])


    cursor.execute("PRAGMA database_list")
    database_info = cursor.fetchall()


    connector.close()


    return db_path, count, database_info




def delete_all_reminders():
    """
    Permanently remove EVERY row from reminders and verify the result.


    secure_delete overwrites deleted SQLite cell content when possible.
    A WAL checkpoint and VACUUM are also attempted so stale deleted content
    is less likely to remain visible in the physical database file.
    """
    db_path = os.path.realpath(dataBase)


    connector = sqlite3.connect(db_path, timeout=10.0)
    cursor = connector.cursor()


    cursor.execute("PRAGMA busy_timeout = 10000")
    cursor.execute("PRAGMA secure_delete = ON")


    cursor.execute("SELECT COUNT(*) FROM reminders")
    before_count = int(cursor.fetchone()[0])


    cursor.execute("DELETE FROM reminders")
    connector.commit()


    # VERIFY the logical table is actually empty.
    cursor.execute("SELECT COUNT(*) FROM reminders")
    remaining_count = int(cursor.fetchone()[0])


    if remaining_count != 0:
        connector.close()
        raise RuntimeError(
            f"Reminder deletion verification failed. "
            f"{remaining_count} row(s) remain in {db_path}"
        )


    # Flush WAL content if this database happens to be using WAL mode.
    try:
        cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connector.commit()
    except sqlite3.DatabaseError:
        pass


    connector.close()


    # Compact the database so deleted rows do not linger in unused pages.
    # Failure here does NOT mean the logical DELETE failed; the row-count
    # verification above is authoritative.
    try:
        vacuum_connection = sqlite3.connect(db_path, timeout=10.0)
        vacuum_connection.execute("PRAGMA busy_timeout = 10000")
        vacuum_connection.execute("VACUUM")
        vacuum_connection.close()
    except sqlite3.DatabaseError as e:
        print("Reminder cleanup VACUUM skipped:", e)


    # Final independent verification using a fresh connection.
    verify_connection = sqlite3.connect(db_path, timeout=10.0)
    verify_cursor = verify_connection.cursor()
    verify_cursor.execute("SELECT COUNT(*) FROM reminders")
    final_count = int(verify_cursor.fetchone()[0])
    verify_connection.close()


    if final_count != 0:
        raise RuntimeError(
            f"Final reminder verification failed. "
            f"{final_count} row(s) remain in {db_path}"
        )


    return before_count, final_count, db_path




def analyze_reminder(user_input):
    """
    Classify reminder creation/deletion requests.


    Output exactly one of:
      CREATE|YYYY-MM-DD HH:MM|reminder text
      DELETE|search text
      DELETE_ALL
      NOTHING
    """
    prompt = f"""
    Determine what reminder action the user is requesting.


    Respond using EXACTLY one of these formats:


    If the user wants to CREATE a reminder:
    CREATE|YYYY-MM-DD HH:MM|reminder text


    If the user wants to CANCEL, REMOVE, DELETE, or FORGET one specific reminder:
    DELETE|short search text that identifies the reminder


    If the user wants to remove ALL reminders, clear reminders, delete every reminder,
    or forget all reminders:
    DELETE_ALL


    If the message is not actually a reminder action:
    NOTHING


    Important:
    - "Remind me to call Mom tomorrow at 3" is CREATE.
    - "Cancel my reminder to call Mom" is DELETE|call Mom.
    - "Delete the reminder about laundry" is DELETE|laundry.
    - "Forget my reminder to buy milk" is DELETE|buy milk.
    - "Clear all my reminders" is DELETE_ALL.
    - Do not turn deletion requests into CREATE requests.


    Current date and time:
    {time.strftime("%Y-%m-%d %H:%M")}


    User message:
    {user_input}
    """


    response = classifier_generate(prompt)
    return response.text.strip()




def format_reminder_time_for_speech(remind_at):
    """Convert the database timestamp into something natural for Piper to say."""
    try:
        parsed = time.strptime(remind_at, "%Y-%m-%d %H:%M")
        spoken = time.strftime("%A, %B %d at %I:%M %p", parsed)
        return spoken.replace(" 0", " ")
    except Exception:
        return remind_at




def listen_for_reminder_confirmation():
    """
    Listen once for a yes/no answer without requiring the wake word again.


    Returns:
        True  -> user confirmed
        False -> user cancelled
        None  -> answer was missing or unclear
    """
    confirmation_audio = pyaudio.PyAudio()
    stream = None


    try:
        mic_device = choose_wake_input_device(confirmation_audio)


        stream = confirmation_audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=MIC_RATE,
            input=True,
            input_device_index=mic_device,
            frames_per_buffer=CHUNK,
        )


        # Give the speaker output a moment to disappear from the microphone.
        time.sleep(0.15)


        # Measure a short local noise floor before waiting for the answer.
        noise_levels = []
        for _ in range(4):
            data = read_input_chunk(stream)
            samples = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
            noise_levels.append(rms)


        noise_floor = float(np.median(noise_levels)) if noise_levels else 50.0
        speech_threshold = max(noise_floor * 2.2, 70.0)


        print(
            f"Reminder confirmation noise floor: {noise_floor:.0f} | "
            f"speech threshold: {speech_threshold:.0f}"
        )
        print("Listening for yes/no confirmation...")


        frames = []
        speech_started = False
        silent_time = 0.0
        waiting_time = 0.0
        chunk_duration = CHUNK / MIC_RATE


        start_timeout = 5.0
        max_answer_time = 4.0
        end_silence = 0.60


        while True:
            data = read_input_chunk(stream)
            samples = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))


            if not speech_started:
                waiting_time += chunk_duration


                if rms >= speech_threshold:
                    speech_started = True
                    frames.append(data)
                    print("Confirmation speech detected.")
                elif waiting_time >= start_timeout:
                    print("No confirmation heard.")
                    return None
            else:
                frames.append(data)


                if rms < speech_threshold:
                    silent_time += chunk_duration
                else:
                    silent_time = 0.0


                if silent_time >= end_silence:
                    break


                if len(frames) * chunk_duration >= max_answer_time:
                    break


        if not frames:
            return None


        raw_audio = b"".join(frames)
        answer_audio = sr.AudioData(
            raw_audio,
            sample_rate=MIC_RATE,
            sample_width=2,
        )


        try:
            answer = recognizer.recognize_google(answer_audio)
        except sr.UnknownValueError:
            print("Could not understand reminder confirmation.")
            return None


        normalized = " ".join(
            answer.lower().strip().replace("-", " ").split()
        )
        print("Reminder confirmation:", answer)


        yes_phrases = {
            "yes",
            "yeah",
            "yep",
            "correct",
            "confirmed",
            "confirm",
            "that's correct",
            "that is correct",
            "do it",
            "save it",
            "sounds good",
            "sure",
        }


        no_phrases = {
            "no",
            "nope",
            "cancel",
            "cancel it",
            "never mind",
            "nevermind",
            "don't",
            "do not",
            "incorrect",
            "that's wrong",
            "that is wrong",
        }


        if normalized in yes_phrases or normalized.startswith("yes "):
            return True


        if normalized in no_phrases or normalized.startswith("no "):
            return False


        print(f"Unclear reminder confirmation: {normalized!r}")
        return None


    except sr.RequestError as e:
        print("Reminder confirmation speech-recognition error:", e)
        return None
    except Exception as e:
        print("Reminder confirmation error:", e)
        return None
    finally:
        if stream is not None:
            try:
                if stream.is_active():
                    stream.stop_stream()
            except Exception:
                pass


            try:
                stream.close()
            except Exception:
                pass


        confirmation_audio.terminate()




def confirm_and_save_reminder(reminder, remind_at):
    """Read a reminder back to the user and save it only after a clear yes."""
    spoken_time = format_reminder_time_for_speech(remind_at)


    speak(
        f"Confirming. Remind you to {reminder} on {spoken_time}. "
        "Is that correct?"
    )


    confirmed = listen_for_reminder_confirmation()


    if confirmed is True:
        save_reminder(reminder, remind_at)
        print(f"Reminder created: {reminder} at {remind_at}")
        speak("Confirmed. The reminder has been saved.")
        return True


    if confirmed is False:
        print("Reminder creation cancelled by user.")
        speak("Cancelled. I did not save the reminder.")
        return True


    print("Reminder creation cancelled because confirmation was unclear.")
    speak(
        "I could not confirm that reminder, so I did not save it. "
        "You can ask me again if you still want it."
    )
    return True




def process_reminder(user_input):
    """
    Execute reminder create/delete actions.


    Returns True when the message was handled as a reminder action.
    """
    result = analyze_reminder(user_input)
    print("Reminder analysis:", result)


    if result.startswith("CREATE|"):
        parts = result.split("|", 2)


        if len(parts) == 3:
            remind_at = parts[1].strip()
            reminder = parts[2].strip()


            # Never write a new reminder until the user confirms the
            # parsed reminder text and date/time out loud.
            return confirm_and_save_reminder(reminder, remind_at)


    if result.startswith("DELETE|"):
        search_text = result.split("|", 1)[1].strip()


        if search_text:
            deleted_count = delete_reminders(search_text)
            print(
                f"Deleted {deleted_count} reminder(s) matching: "
                f"{search_text}"
            )
            return True


    if result == "DELETE_ALL":
        try:
            deleted_count, remaining_count, db_path = delete_all_reminders()


            print("========================================")
            print("        REMINDER DATABASE CLEARED")
            print("========================================")
            print(f"Database: {db_path}")
            print(f"Rows removed: {deleted_count}")
            print(f"Rows remaining: {remaining_count}")


            if remaining_count == 0:
                speak("All reminders have been deleted.")
                return True


        except Exception as e:
            print("Reminder deletion failed:", e)
            speak(
                "I was unable to verify that the reminders were deleted."
            )
            return True


    return False




def get_memories():
    connector = sqlite3.connect(dataBase)
    cursor = connector.cursor()


    cursor.execute('''
       SELECT category, memory
       FROM memory
       ORDER BY created_at DESC
       ''')


    memory = cursor.fetchall()
    connector.close()
    return memory   


def get_memory_context():
   memories = get_memories()


   if not memories:
       return "No memories found." \


   memoryText = ""


   for category, memory in memories:
       memoryText += f"{category}: {memory}\n"


   return memoryText


def analyze_memory(user_input):
    """
    Decide whether the user's message contains a genuinely persistent memory.


    IMPORTANT:
    Reminders, alarms, scheduled tasks, dates/times to do something, and other
    temporary obligations belong ONLY in the reminders table and must NEVER
    be stored as permanent memory.
    """
    prompt = f"""
    Determine whether the following user message contains information that
    should be permanently remembered about the user.


    Good permanent memories include:
    - stable preferences
    - long-term goals
    - ongoing project facts
    - personal facts the user explicitly asks to remember


    NEVER save any of the following as permanent memory:
    - reminders
    - reminder requests
    - alarms
    - scheduled tasks
    - appointments
    - temporary deadlines
    - instructions such as "remind me to..."
    - anything whose main purpose is to happen at a particular date or time


    If something should be remembered, respond EXACTLY:
    SAVE|category|memory


    Otherwise respond EXACTLY:
    NOTHING


    User message:
    {user_input}
    """


    response = classifier_generate(prompt)
    return response.text.strip()




def process_memory(user_input):
    """
    Save a permanent memory, with a hard reminder-exclusion safety check.
    """
    lower_input = user_input.lower()


    # Even if the model classifier makes a mistake, reminder-like requests
    # are never allowed into the memory table.
    if "remind me" in lower_input or "reminder" in lower_input:
        print("Memory save skipped: reminder content belongs in reminders table.")
        return False


    result = analyze_memory(user_input)
    print("Memory Analysis Result:", result)


    if result.startswith("SAVE|"):
        parts = result.split("|", 2)


        if len(parts) == 3:
            category = parts[1].strip()
            memory = parts[2].strip()


            if "remind" in category.lower():
                print("Memory save rejected: classifier returned reminder category.")
                return False


            save_memory(category, memory)
            print(f"Memory saved: Category: {category}, Memory: {memory}")
            return True


    return False




def needs_memory(user_input):
   memory_keywords = [
       "remeember",
       "forgot",
       "forget",
       "favorite",
       "remind",
       "log",
       "remind me",
       "i like",
       "my goal",
       "save this",
       "log this"
   ]


   user_lower = user_input.lower()
   return any(keyword in user_lower for keyword in memory_keywords)




def list_alsa_playback_devices():
    """
    Parse `aplay -l` and return real hardware playback devices.


    We retain the ALSA card ID as well as the numeric index so fallback
    playback can use a stable name such as:
        plughw:CARD=MAX98357A,DEV=0
    instead of a fragile number such as:
        plughw:2,0
    """
    try:
        result = subprocess.run(
            ["aplay", "-l"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception as e:
        print("Could not query ALSA playback devices:", e)
        return []


    output = (result.stdout or "") + "\n" + (result.stderr or "")
    devices = []


    # Typical line:
    # card 1: MAX98357A [MAX98357A], device 0: ...
    pattern = re.compile(
        r"card\s+(\d+):\s+([^\s]+)\s+\[([^\]]*)\],\s+"
        r"device\s+(\d+):\s+([^\[]+)\[([^\]]*)\]",
        re.IGNORECASE,
    )


    for match in pattern.finditer(output):
        card_index = int(match.group(1))
        card_id = match.group(2).strip()
        card_name = match.group(3).strip()
        device = int(match.group(4))
        pcm_name = match.group(5).strip()
        pcm_desc = match.group(6).strip()


        description = " ".join(
            part for part in (card_id, card_name, pcm_name, pcm_desc) if part
        )


        lower = description.lower()
        score = 0


        # Typical Raspberry Pi digital-audio / I2S speaker devices.
        preferred_terms = (
            "i2s",
            "max98357",
            "hifiberry",
            "dac",
            "speaker",
            "wm8960",
            "simple",
            "sndrpihifi",
            "audioinjector",
        )


        # Avoid choosing obvious non-speaker devices when we need a fallback.
        avoided_terms = (
            "hdmi",
            "usb pnp sound device",
            "microphone",
        )


        if any(term in lower for term in preferred_terms):
            score += 100


        if any(term in lower for term in avoided_terms):
            score -= 100


        devices.append(
            {
                "card_index": card_index,
                "card_id": card_id,
                "device": device,
                "description": description,
                "named_alsa": f"plughw:CARD={card_id},DEV={device}",
                "score": score,
            }
        )


    devices.sort(key=lambda item: item["score"], reverse=True)
    return devices




def run_aplay(wav_path, device=None):
    """
    Play through ALSA.


    device=None intentionally runs plain:
        aplay file.wav
    so ALSA uses the Pi's configured default output.
    """
    command = ["aplay"]


    if device is not None:
        command.extend(["-D", device])


    command.append(wav_path)


    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=120,
    )




AUDIO_OUTPUT_DEVICE = None
AUDIO_OUTPUT_INITIALIZED = False




def play_wav_file(wav_path):
    """
    Use the system's normal ALSA output first.


    This is the most reliable choice when the user's I2S/I2C audio device
    already works elsewhere on the Pi. If ALSA default fails, try named
    hardware devices discovered from `aplay -l`.
    """
    global AUDIO_OUTPUT_DEVICE
    global AUDIO_OUTPUT_INITIALIZED


    # First choice: do NOT supply -D. Respect the working ALSA default route.
    if not AUDIO_OUTPUT_INITIALIZED:
        print("Speaker output: ALSA default")
        AUDIO_OUTPUT_INITIALIZED = True
        AUDIO_OUTPUT_DEVICE = None


    result = run_aplay(wav_path, AUDIO_OUTPUT_DEVICE)


    if result.returncode == 0:
        return True


    print(
        "Default audio playback failed:",
        (result.stderr or result.stdout).strip(),
    )


    # If the default route fails, try stable card-name fallbacks.
    devices = list_alsa_playback_devices()


    if not devices:
        print("No ALSA hardware playback devices were found by `aplay -l`.")
        return False


    for device in devices:
        candidate = device["named_alsa"]


        print(
            f"Trying speaker fallback: {candidate} "
            f"({device['description']})"
        )


        retry = run_aplay(wav_path, candidate)


        if retry.returncode == 0:
            AUDIO_OUTPUT_DEVICE = candidate
            print(f"Speaker output selected: {candidate}")
            return True


        print(
            f"Playback failed on {candidate}: "
            f"{(retry.stderr or retry.stdout).strip()}"
        )


    # Go back to probing ALSA default next time in case devices changed.
    AUDIO_OUTPUT_DEVICE = None
    AUDIO_OUTPUT_INITIALIZED = False


    return False






def _scale_pcm16(audio_bytes, volume):
    """Scale signed 16-bit PCM without launching ffmpeg."""
    if volume == 1.0:
        return audio_bytes


    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    samples *= float(volume)
    samples = np.clip(samples, -32768, 32767).astype(np.int16)
    return samples.tobytes()




def _speak_file_fallback_unlocked(text):
    """
    Reliable fallback using the older WAV -> ffmpeg -> aplay path.
    Called only while tts_lock is already held.
    """
    output_file = "/tmp/glados_response.wav"
    quiet_file = "/tmp/glados_quiet.wav"


    try:
        with wave.open(output_file, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)


        if (
            not os.path.exists(output_file)
            or os.path.getsize(output_file) == 0
        ):
            print("TTS error: Piper produced an empty WAV file.")
            return False


        ffmpeg_result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                output_file,
                "-filter:a",
                f"volume={PIPER_VOLUME}",
                quiet_file,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )


        if ffmpeg_result.returncode != 0:
            print(
                "ffmpeg audio processing failed:",
                ffmpeg_result.stderr.strip(),
            )
            return False


        return play_wav_file(quiet_file)


    except Exception as e:
        print("Fallback TTS/playback error:", e)
        return False




def _start_raw_aplay(sample_rate, channels):
    """
    Open ALSA for raw 16-bit PCM streaming.


    If a named output was previously discovered, reuse it. Otherwise use
    ALSA's configured default output, which is already working on this Pi.
    """
    command = ["aplay", "-q"]


    if AUDIO_OUTPUT_DEVICE is not None:
        command.extend(["-D", AUDIO_OUTPUT_DEVICE])


    command.extend(
        [
            "-t", "raw",
            "-f", "S16_LE",
            "-r", str(int(sample_rate)),
            "-c", str(int(channels)),
        ]
    )


    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )




def _speak_impl(text):
    """
    Low-latency Piper playback.


    Piper synthesizes sentence/audio chunks and they are immediately piped
    into ALSA. This removes the normal wait for:
        entire WAV synthesis -> ffmpeg process -> final aplay process


    If streaming is unavailable for any reason, the old file-based method
    is used automatically.
    """
    with tts_lock:
        if not ENABLE_STREAMING_TTS:
            return _speak_file_fallback_unlocked(text)


        tts_start = time.monotonic()
        process = None


        try:
            audio_stream = iter(voice.synthesize(text))
            first_chunk = next(audio_stream, None)


            if first_chunk is None:
                print("Streaming TTS produced no audio.")
                return False


            first_chunk_time = time.monotonic() - tts_start


            if PRINT_LATENCY_TIMINGS:
                print(
                    f"Latency | Piper first audio chunk: "
                    f"{first_chunk_time:.2f}s"
                )


            process = _start_raw_aplay(
                first_chunk.sample_rate,
                first_chunk.sample_channels,
            )


            if process.stdin is None:
                raise RuntimeError("Could not open aplay stdin.")


            # Send the first chunk immediately.
            process.stdin.write(
                _scale_pcm16(
                    first_chunk.audio_int16_bytes,
                    PIPER_VOLUME,
                )
            )
            process.stdin.flush()


            # Continue synthesizing/playing subsequent chunks while the
            # previous audio is already coming out of the speakers.
            for chunk in audio_stream:
                process.stdin.write(
                    _scale_pcm16(
                        chunk.audio_int16_bytes,
                        PIPER_VOLUME,
                    )
                )
                process.stdin.flush()


            process.stdin.close()
            process.stdin = None


            # A wedged ALSA process must not freeze the entire assistant.
            # Allow ample time for long spoken answers, but keep it bounded.
            playback_timeout = max(20.0, min(180.0, 12.0 + len(text) * 0.18))
            return_code = process.wait(timeout=playback_timeout)


            if return_code != 0:
                error_text = ""
                if process.stderr is not None:
                    error_text = process.stderr.read().decode(
                        errors="replace"
                    ).strip()


                raise RuntimeError(
                    f"Streaming aplay failed: {error_text}"
                )


            if PRINT_LATENCY_TIMINGS:
                print(
                    f"Latency | Piper synthesis/playback total: "
                    f"{time.monotonic() - tts_start:.2f}s"
                )


            return True


        except Exception as e:
            print(
                "Streaming TTS unavailable; using WAV fallback:",
                e,
            )


            try:
                if process is not None and process.poll() is None:
                    process.terminate()
            except Exception:
                pass


            return _speak_file_fallback_unlocked(text)






def speak(text):
    """
    Public speech wrapper.


    YOLO may remain active while this flag is set, but physical pan motion
    is suppressed by the vision loop until speech playback finishes.
    """
    assistant_speaking_event.set()


    try:
        return _speak_impl(text)
    finally:
        assistant_speaking_event.clear()




def cleanup_legacy_reminder_memories():
    """
    Remove reminder rows accidentally stored in the permanent memory table by
    older versions of this program.


    This intentionally targets reminder-labeled categories only, so normal
    permanent memories are not broadly pattern-deleted.
    """
    db_path = os.path.realpath(dataBase)


    connector = sqlite3.connect(db_path, timeout=10.0)
    cursor = connector.cursor()


    cursor.execute("PRAGMA busy_timeout = 10000")
    cursor.execute("PRAGMA secure_delete = ON")


    cursor.execute(
        """
        DELETE FROM memory
        WHERE LOWER(COALESCE(category, '')) LIKE '%remind%'
        """
    )


    deleted = cursor.rowcount
    connector.commit()
    connector.close()


    return deleted




def delete_memories(search_text):
    """
    Permanently delete specific memories by matching either category or memory
    text. Matching is case-insensitive.
    """
    db_path = os.path.realpath(dataBase)


    connector = sqlite3.connect(db_path, timeout=10.0)
    cursor = connector.cursor()


    cursor.execute("PRAGMA busy_timeout = 10000")
    cursor.execute("PRAGMA secure_delete = ON")


    pattern = f"%{search_text}%"


    cursor.execute(
        """
        DELETE FROM memory
        WHERE memory LIKE ? COLLATE NOCASE
           OR category LIKE ? COLLATE NOCASE
        """,
        (pattern, pattern),
    )


    deleted = cursor.rowcount
    connector.commit()
    connector.close()


    return deleted




def delete_all_memories():
    """
    Permanently clear the permanent-memory table and verify that zero rows
    remain before success is reported.
    """
    db_path = os.path.realpath(dataBase)


    connector = sqlite3.connect(db_path, timeout=10.0)
    cursor = connector.cursor()


    cursor.execute("PRAGMA busy_timeout = 10000")
    cursor.execute("PRAGMA secure_delete = ON")


    cursor.execute("SELECT COUNT(*) FROM memory")
    before_count = int(cursor.fetchone()[0])


    cursor.execute("DELETE FROM memory")
    connector.commit()


    cursor.execute("SELECT COUNT(*) FROM memory")
    remaining = int(cursor.fetchone()[0])


    if remaining != 0:
        connector.close()
        raise RuntimeError(
            f"Memory deletion verification failed. "
            f"{remaining} row(s) remain in {db_path}"
        )


    connector.close()


    # Compact deleted pages where possible.
    try:
        vacuum_connection = sqlite3.connect(db_path, timeout=10.0)
        vacuum_connection.execute("PRAGMA busy_timeout = 10000")
        vacuum_connection.execute("VACUUM")
        vacuum_connection.close()
    except sqlite3.DatabaseError as e:
        print("Memory cleanup VACUUM skipped:", e)


    # Fresh-connection verification.
    verify = sqlite3.connect(db_path, timeout=10.0)
    verify_cursor = verify.cursor()
    verify_cursor.execute("SELECT COUNT(*) FROM memory")
    final_count = int(verify_cursor.fetchone()[0])
    verify.close()


    if final_count != 0:
        raise RuntimeError(
            f"Final memory verification failed. "
            f"{final_count} row(s) remain in {db_path}"
        )


    return before_count, final_count, db_path




def analyze_forget(user_input):
    """
    Classify permanent-memory deletion requests.


    Output exactly one of:
      DELETE|literal search text
      DELETE_ALL
      NOTHING
    """
    prompt = f"""
    Determine whether the user wants GLaDOS to delete PERMANENT MEMORY.


    Respond EXACTLY with one of:


    DELETE|search text
    DELETE_ALL
    NOTHING


    Rules:
    - "Forget that I like chocolate" -> DELETE|chocolate
    - "Delete the memory that I like jazz" -> DELETE|jazz
    - "Forget my favorite color" -> DELETE|favorite color
    - "Delete all memories" -> DELETE_ALL
    - "Forget everything you remember about me" -> DELETE_ALL
    - "Clear your memory" -> DELETE_ALL


    IMPORTANT:
    Reminder deletion is handled somewhere else. If the user is specifically
    talking about a reminder, respond NOTHING here.


    The DELETE search text should be a SHORT LITERAL phrase likely to actually
    appear in the stored category or memory. Do not replace it with an unrelated
    semantic label.


    User message:
    {user_input}
    """


    response = classifier_generate(prompt)
    return response.text.strip()




def process_forget(user_input):
    """
    Execute permanent-memory deletion.


    Returns True when a memory deletion request was handled.
    """
    result = analyze_forget(user_input)
    print("Memory deletion analysis:", result)


    if result == "DELETE_ALL":
        try:
            deleted_count, remaining_count, db_path = delete_all_memories()


            print("========================================")
            print("          MEMORY DATABASE CLEARED")
            print("========================================")
            print(f"Database: {db_path}")
            print(f"Rows removed: {deleted_count}")
            print(f"Rows remaining: {remaining_count}")


            if remaining_count == 0:
                speak("All permanent memories have been deleted.")
                return True


        except Exception as e:
            print("Memory deletion failed:", e)
            speak("I was unable to verify that the memories were deleted.")
            return True


    if result.startswith("DELETE|"):
        search_text = result.split("|", 1)[1].strip()


        if search_text:
            deleted_count = delete_memories(search_text)


            print(
                f"Deleted {deleted_count} permanent memory row(s) matching: "
                f"{search_text}"
            )


            if deleted_count > 0:
                speak("That memory has been deleted.")
            else:
                speak("I could not find a matching stored memory.")


            return True


    return False




initialize_database()


try:
    _legacy_reminder_memories = cleanup_legacy_reminder_memories()
    if _legacy_reminder_memories:
        print(
            f"Removed {_legacy_reminder_memories} legacy reminder "
            f"row(s) from permanent memory."
        )
except Exception as e:
    print("Legacy reminder-memory cleanup error:", e)




try:
    _db_path, _reminder_count, _db_info = get_reminder_database_status()
    print(f"Reminder database: {_db_path}")
    print(f"Reminder rows currently stored: {_reminder_count}")
except Exception as e:
    print("Reminder database status error:", e)


wled_color(255, 0, 0)
time.sleep(3)
wled_off()










memories = get_memories()




def choose_wake_input_device(audio):
    """
    Return a valid mono input device.


    The previous code assumed PyAudio device index 1 was always the
    microphone. Device indexes can change between boots or when USB/audio
    devices enumerate differently. If the preferred device is not a usable
    input, fall back to the current default input, then any valid input.
    """


    def usable(device_index):
        try:
            info = audio.get_device_info_by_index(device_index)


            if int(info.get("maxInputChannels", 0)) < CHANNELS:
                return False


            # Also verify that this device accepts our actual wake-word format.
            audio.is_format_supported(
                MIC_RATE,
                input_device=device_index,
                input_channels=CHANNELS,
                input_format=FORMAT,
            )
            return True


        except Exception:
            return False


    # Keep using the user's original preferred index when it is genuinely valid.
    if usable(PREFERRED_WAKE_MIC_DEVICE):
        info = audio.get_device_info_by_index(PREFERRED_WAKE_MIC_DEVICE)
        print(
            f"Wake microphone: device {PREFERRED_WAKE_MIC_DEVICE} "
            f"({info.get('name', 'unknown')})"
        )
        return PREFERRED_WAKE_MIC_DEVICE


    # Device numbering often changes. Prefer PyAudio's current default input.
    try:
        default_info = audio.get_default_input_device_info()
        default_index = int(default_info["index"])


        if usable(default_index):
            print(
                f"Preferred microphone device {PREFERRED_WAKE_MIC_DEVICE} "
                f"is not a valid mono input."
            )
            print(
                f"Wake microphone fallback: device {default_index} "
                f"({default_info.get('name', 'unknown')})"
            )
            return default_index
    except Exception:
        pass


    # Last fallback: search all PortAudio devices for a working mono input.
    for device_index in range(audio.get_device_count()):
        if usable(device_index):
            info = audio.get_device_info_by_index(device_index)
            print(
                f"Preferred microphone device {PREFERRED_WAKE_MIC_DEVICE} "
                f"is unavailable."
            )
            print(
                f"Wake microphone fallback: device {device_index} "
                f"({info.get('name', 'unknown')})"
            )
            return device_index


    # Give a useful diagnostic instead of PortAudio's vague channel error.
    discovered = []


    for device_index in range(audio.get_device_count()):
        try:
            info = audio.get_device_info_by_index(device_index)
            discovered.append(
                f"{device_index}: {info.get('name', 'unknown')} "
                f"(inputs={int(info.get('maxInputChannels', 0))})"
            )
        except Exception:
            pass


    raise RuntimeError(
        "No PyAudio input device supports "
        f"{CHANNELS} channel at {MIC_RATE} Hz. Devices: "
        + "; ".join(discovered)
    )




def read_input_chunk(stream, frames=CHUNK, timeout=MIC_CHUNK_TIMEOUT):
    """
    Read one microphone chunk without allowing PortAudio to block forever.


    PyAudio's normal stream.read() is blocking. If the USB/I2S audio stack
    stalls, the old code could remain inside that call indefinitely while
    assistant_busy_event stayed set. Polling get_read_available() gives us a
    watchdog and lets the assistant recover/reopen the microphone instead.
    """
    deadline = time.monotonic() + timeout


    while time.monotonic() < deadline:
        if shutdown_event.is_set():
            raise RuntimeError("GLaDOS is shutting down.")


        try:
            if not stream.is_active():
                stream.start_stream()


            if stream.get_read_available() >= frames:
                return stream.read(
                    frames,
                    exception_on_overflow=False,
                )
        except OSError as e:
            raise RuntimeError(f"Microphone stream error: {e}") from e


        time.sleep(0.01)


    raise TimeoutError(
        f"Microphone produced no audio for {timeout:.1f} seconds."
    )




def listen_for_wake_and_command():


   wake_audio = pyaudio.PyAudio()
   stream = None


   recent_levels = []


   # Resolve a valid microphone instead of assuming device index 1.
   try:
       wake_mic_device = choose_wake_input_device(wake_audio)


       stream = wake_audio.open(
           format=FORMAT,
           channels=CHANNELS,
           rate=MIC_RATE,
           input=True,
           input_device_index=wake_mic_device,
           frames_per_buffer=CHUNK
       )


       # Start every wake attempt with a clean model state.
       wake_model.reset()


       print("Waiting for wake word...")


       while True:


           audio_data = read_input_chunk(stream)


           audio_array = np.frombuffer(
               audio_data,
               dtype=np.int16
           )


           # Keep track of recent microphone levels.
           rms = np.sqrt(
               np.mean(
                   audio_array.astype(np.float32) ** 2
               )
           )


           recent_levels.append(rms)


           if len(recent_levels) > 50:
               recent_levels.pop(0)


           # Resample 44.1 kHz -> 16 kHz for OpenWakeWord
           resampled_audio = resample_poly(
               audio_array.astype(np.float32),
               160,
               441
           )


           resampled_audio = np.clip(
               resampled_audio,
               -32768,
               32767
           ).astype(np.int16)


           if len(resampled_audio) > 1280:
               resampled_audio = resampled_audio[:1280]


           elif len(resampled_audio) < 1280:
               resampled_audio = np.pad(
                   resampled_audio,
                   (0, 1280 - len(resampled_audio))
               )


           prediction = wake_model.predict(
               resampled_audio
           )


           #provides the confidence score of the wakeword, so whenever I say hey glados basically
           score = float(
               prediction["hey_glados"]
           )


           if score >= 0.05:
               print(
                   f"Wake word score: {score:.3f}"
               )




           #the confidence value that the wakeword must detect in order for GLaDOS to start listening
           if score >= WAKE_THRESHOLD:


               print(
                   f"Wake word detected! "
                   f"Score: {score:.2f}"
               )


               wake_model.reset()


               # Freeze visual tracking while GLaDOS is interacting.
               # This frees Pi CPU for speech recognition and TTS, and also
               # prevents servo movement/noise while the user is speaking.
               assistant_busy_event.set()


               # If GLaDOS was sleeping, this same wake word brings her back
               # online. Keep standby set until after the spoken acknowledgement
               # so YOLO/capture work and servo movement remain suspended.
               waking_from_standby = standby_event.is_set()
               acknowledgement = (
                   "Online. Listening."
                   if waking_from_standby
                   else WAKE_ACK_TEXT
               )


               # Pause the microphone while GLaDOS acknowledges the wake word
               # so her own voice is not recorded as part of the user's command.
               try:
                   if stream is not None and stream.is_active():
                       stream.stop_stream()


                   speak(acknowledgement)


               finally:
                   if waking_from_standby:
                       standby_event.clear()
                       print("GLaDOS resumed from standby.")


                   if stream is not None and not stream.is_active():
                       stream.start_stream()


               # Give the speakers/microphone a brief moment to settle before
               # beginning command detection.
               time.sleep(WAKE_ACK_SETTLE_TIME)


               break




       if recent_levels:
           noise_floor = float(
               np.percentile(
                   recent_levels,
                   25
               )
           )


       else:
           noise_floor = 50.0


       speech_threshold = max(
           noise_floor * 2.2,
           70
       )


       print(
           f"Noise floor: {noise_floor:.0f} | "
           f"Speech threshold: {speech_threshold:.0f}"
       )


       print("Listening for command...")


       #GLaDOS will detect whenever you start speaking, and whever you stop speaking


       command_frames = []


       speech_started = False
       silent_time = 0.0
       waiting_time = 0.0


       chunk_duration = CHUNK / MIC_RATE


       #can talk for a max value of whatever you set here
       START_TIMEOUT = 5.0
       MAX_COMMAND_TIME = 10.0
       END_SILENCE = 0.85


       while True:


           data = read_input_chunk(stream)


           samples = np.frombuffer(
               data,
               dtype=np.int16
           )


           rms = np.sqrt(
               np.mean(
                   samples.astype(np.float32) ** 2
               )
           )




           if not speech_started:


               waiting_time += chunk_duration


               if rms >= speech_threshold:


                   speech_started = True


                   print("Speech detected.")


                   command_frames.append(data)


               elif waiting_time >= START_TIMEOUT:


                   print("No command heard.")
                   assistant_busy_event.clear()


                   return None






           else:


               command_frames.append(data)


               if rms < speech_threshold:
                   silent_time += chunk_duration


               else:
                   silent_time = 0.0


               # Stop after enough silence.
               if silent_time >= END_SILENCE:


                   print("Command captured.")


                   break


               #limit so it can never
               # listen forever
               recorded_time = (
                   len(command_frames)
                   * chunk_duration
               )


               if recorded_time >= MAX_COMMAND_TIME:


                   print(
                       "Maximum command length reached."
                   )


                   break






       raw_audio = b"".join(
           command_frames
       )


       user_audio = sr.AudioData(
           raw_audio,
           sample_rate=MIC_RATE,
           sample_width=2
       )


       # Save exactly what Google receives.
       with open(
           "/tmp/last_command.wav",
           "wb"
       ) as f:


           f.write(
               user_audio.get_wav_data()
           )


       print("Recognizing...")


       recognition_start = time.monotonic()


       user_input = recognizer.recognize_google(
           user_audio
       )


       if PRINT_LATENCY_TIMINGS:
           print(
               f"Latency | Speech recognition: "
               f"{time.monotonic() - recognition_start:.2f}s"
           )


       print(
           "You:",
           user_input
       )


       return user_input


   finally:


       if stream is not None:


           if stream.is_active():
               stream.stop_stream()


           stream.close()


       wake_audio.terminate()






# ============================================================
# CONTINUOUS VISION / SERVO SYSTEM
# ============================================================


def track_person_closed_loop(
    picam2,
    vision_model,
    pan_motion,
    initial_error_sign,
    initial_person_cx,
):
    """
    Fixed-direction, fresh-frame visual servoing.


    IMPORTANT:
    The servo direction mapping is NEVER changed automatically.
    FLIP_PAN is the single source of truth.


    Every YOLO frame grants only a small, short-lived movement command.
    If a new frame does not arrive quickly, motion expires automatically.
    If the person moves farther from image center, motion stops immediately.
    """
    start_angle = pan_motion.get_angle()
    previous_cx = initial_person_cx
    last_abs_error = abs(
        initial_person_cx - (CAMERA_WIDTH / 2.0)
    )


    fixed_multiplier = -1.0 if FLIP_PAN else 1.0


    print(
        f"Fixed pan mapping active: FLIP_PAN={FLIP_PAN}, "
        f"multiplier={fixed_multiplier:+.0f}"
    )


    while not shutdown_event.is_set():


        
        if standby_event.is_set():
            pan_motion.disengage()
            pan_motion.save()
            return pan_motion.get_angle()


         
        if assistant_busy_event.is_set() or assistant_speaking_event.is_set():
            pan_motion.hold(emergency=True)
            pan_motion.save()
            return pan_motion.get_angle()


        frame = picam2.capture_array()
        _, frame_width = frame.shape[:2]


        person = detect_largest_person(
            vision_model,
            frame,
        )


        if person is None:
            # No fresh visual target = no movement.
            pan_motion.hold(emergency=True)
            pan_motion.save()
            print("Tracking stopped: person lost.")
            return pan_motion.get_angle()


        x1, y1, x2, y2 = person
        person_cx = (x1 + x2) / 2.0
        screen_cx = frame_width / 2.0


        error_x = person_cx - screen_cx
        abs_error = abs(error_x)


        # Target is safely inside the desired zone.
        if abs_error <= DEADZONE_PX:
            pan_motion.hold(emergency=True)
            pan_motion.save()
            return pan_motion.get_angle()


        current_sign = 1 if error_x > 0 else -1


        # If the target crossed center, stop rather than reversing during the
        # same movement event.
        if current_sign != initial_error_sign:
            pan_motion.hold(emergency=True)
            pan_motion.save()
            print("Tracking stopped: target crossed center.")
            return pan_motion.get_angle()


        # A large apparent center jump usually means YOLO changed targets.
        if abs(person_cx - previous_cx) > MAX_CONFIRMATION_STEP_PX:
            pan_motion.hold(emergency=True)
            pan_motion.save()
            print(
                "Tracking stopped: sudden YOLO center jump "
                f"{person_cx - previous_cx:+.1f} px"
            )
            return pan_motion.get_angle()


        previous_cx = person_cx


        # The only acceptable result of a pan command is that the person's
        # center gets closer to the image center.
        if (
            last_abs_error is not None
            and abs_error
            > last_abs_error + TRACK_WORSEN_TOLERANCE_PX
        ):
            pan_motion.hold(emergency=True)
            pan_motion.save()


            print(
                "Tracking stopped immediately: image error worsened "
                f"from {last_abs_error:.1f}px to {abs_error:.1f}px. "
                "Direction mapping was NOT changed."
            )


            return pan_motion.get_angle()


        last_abs_error = abs_error


        # Edge protection: do not authorize another command if the target is
        # already dangerously close to leaving the image.
        near_left_edge = x1 <= TRACK_EDGE_GUARD_PX
        near_right_edge = x2 >= (
            frame_width - TRACK_EDGE_GUARD_PX
        )


        if near_left_edge or near_right_edge:
            # Still allow movement only if the target is clearly being driven
            # toward center. The worsening check above already guarantees that.
            # Use an extra-small per-frame movement budget here.
            edge_limited = True
        else:
            edge_limited = False


        event_travel = abs(
            pan_motion.get_angle() - start_angle
        )


        if event_travel >= TRACK_MAX_EVENT_DEG:
            pan_motion.hold(emergency=True)
            pan_motion.save()
            print("Tracking event travel safety limit reached.")
            return pan_motion.get_angle()


        usable_error_range = max(
            1.0,
            (frame_width / 2.0) - DEADZONE_PX,
        )


        normalized_error = clamp(
            (abs_error - DEADZONE_PX)
            / usable_error_range,
            0.0,
            1.0,
        )


        
        speed = (
            TRACK_MIN_SPEED_DEG_PER_SEC
            + (
                TRACK_MAX_SPEED_DEG_PER_SEC
                - TRACK_MIN_SPEED_DEG_PER_SEC
            )
            * (normalized_error ** 0.65)
        )


        image_direction = (
            1.0 if error_x > 0 else -1.0
        )


        # FIXED mapping. No auto-probe, no runtime inversion.
        servo_direction = (
            image_direction * fixed_multiplier
        )


        # Each fresh YOLO result permits only a small amount of travel.
        # This is intentionally conservative near center and near frame edges.
        command_travel_budget = (
            1.0 + 2.0 * normalized_error
        )


        if edge_limited:
            command_travel_budget = min(
                command_travel_budget,
                1.25,
            )


        pan_motion.set_velocity(
            servo_direction * speed,
            max_travel_deg=command_travel_budget,
        )


    pan_motion.hold(emergency=True)
    pan_motion.save()
    return pan_motion.get_angle()




def vision_tracking_loop():
    global pan_angle


    picam2 = None
    pan_servo = None
    pan_motion = None
    confirmation_centers = []


    print()
    print("========================================")
    print("       GLaDOS VISION TRACKER")
    print("          FIXED-DIRECTION FRAME-GUARDED MODE")
    print("========================================")
    print()


    try:
        print(f"Loading YOLO: {MODEL_PATH}")
        vision_model = YOLO(MODEL_PATH)
        print("YOLO loaded.")


        print("Starting camera...")
        picam2 = Picamera2()


        config = picam2.create_preview_configuration(
            main={
                "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
                "format": "RGB888",
            },
            raw={
                "size": (SENSOR_WIDTH, SENSOR_HEIGHT),
            },
            controls={
                "FrameRate": CAMERA_FPS,
            },
            buffer_count=4,
        )


        picam2.configure(config)
        picam2.start()
        camera_running = True
        time.sleep(1.5)


        test_frame = picam2.capture_array()
        actual_height, actual_width = test_frame.shape[:2]


        print(
            f"Camera output: {actual_width}x{actual_height} "
            f"from full-FOV {SENSOR_WIDTH}x{SENSOR_HEIGHT} sensor mode"
        )
        print(f"YOLO inference size: {YOLO_IMAGE_SIZE}")


        print("Initializing GPIO12 hardware PWM...")
        print(f"Kernel: {platform.release()}")
        pan_servo = HardwarePanServo()


        # IMPORTANT: Do not command the pan servo on startup.
        # We only restore the software's last-known angle estimate.
        pan_angle = load_pan_state()
        pan_servo.stop()


        # The motion controller starts without touching the servo.
        # GPIO12 remains silent until the first actual tracking movement.
        pan_motion = SmoothPanMotionController(
            pan_servo,
            pan_angle,
        )


        print(
            f"Pan servo ready; NO startup movement. "
            f"Assumed current angle: {pan_angle:.1f} deg"
        )




        print("Headless vision tracking ready.")


        while not shutdown_event.is_set():
          
            if standby_event.is_set():
                confirmation_centers.clear()


                if pan_motion is not None:
                    pan_motion.disengage()
                    pan_motion.save()
                    pan_angle = pan_motion.get_angle()


                print("GLaDOS vision suspended for standby.")


                # No capture_array() and no YOLO happen while this loop waits.
                while standby_event.is_set() and not shutdown_event.is_set():
                    shutdown_event.wait(0.10)


                if shutdown_event.is_set():
                    break


                print("GLaDOS vision resumed.")
                continue


            if assistant_busy_event.is_set():
                confirmation_centers.clear()
                if pan_motion is not None:
                    pan_motion.hold()
                    pan_angle = pan_motion.get_angle()


                time.sleep(0.01)
                continue


            frame = picam2.capture_array()
            _, frame_width = frame.shape[:2]


            person = detect_largest_person(vision_model, frame)
        
            if assistant_speaking_event.is_set():
                confirmation_centers.clear()


                if pan_motion is not None:
                    pan_motion.hold()
                    pan_angle = pan_motion.get_angle()


                continue




            if person is None:
                confirmation_centers.clear()


                if pan_motion is not None:
                    pan_motion.hold()
                    pan_angle = pan_motion.get_angle()


                continue


            x1, y1, x2, y2 = person
            person_cx = (x1 + x2) / 2.0
            screen_cx = frame_width / 2.0
            error_x = person_cx - screen_cx


            if abs(error_x) <= DEADZONE_PX:
                confirmation_centers.clear()


                if pan_motion is not None:
                    pan_motion.hold()
                    pan_angle = pan_motion.get_angle()


                continue


            # Reject a sudden detection jump before it can influence a sweep.
            # A large jump usually means YOLO briefly selected a false box or
            # switched to another person.
            if confirmation_centers:
                previous_cx = confirmation_centers[-1]


                if abs(person_cx - previous_cx) > MAX_CONFIRMATION_STEP_PX:
                    print(
                        f"Tracking confirmation reset: sudden center jump "
                        f"{person_cx - previous_cx:+.1f} px"
                    )
                    confirmation_centers.clear()


            confirmation_centers.append(person_cx)


            if len(confirmation_centers) > REQUIRED_CONFIRMATIONS:
                confirmation_centers.pop(0)


            confirmation_count = len(confirmation_centers)


            if confirmation_count < REQUIRED_CONFIRMATIONS:
                continue


            spread = max(confirmation_centers) - min(confirmation_centers)


            if spread > MAX_CONFIRMATION_SPREAD_PX:
                newest = confirmation_centers[-1]
                confirmation_centers.clear()
                confirmation_centers.append(newest)
                continue


            # Every confirmation must agree about which side of center the
            # person is on. This prevents a left/right detection switch from
            # becoming one large incorrect sweep.
            all_left = all(
                cx < (screen_cx - DEADZONE_PX)
                for cx in confirmation_centers
            )
            all_right = all(
                cx > (screen_cx + DEADZONE_PX)
                for cx in confirmation_centers
            )


            if not (all_left or all_right):
                confirmation_centers.clear()
                continue


            # Median is much more resistant to one weird YOLO box than mean.
            confirmed_cx = float(np.median(confirmation_centers))
            confirmation_centers.clear()


            confirmed_error_px = confirmed_cx - screen_cx
            initial_error_sign = 1 if confirmed_error_px > 0 else -1


            print()
            print("========================================")
            print("       CLOSED-LOOP GLaDOS TRACK")
            print("========================================")
            print(f"Initial pixel error: {confirmed_error_px:+.1f} px")
            print(f"Pan start: {pan_motion.get_angle():.1f} deg")


            # ONE continuous movement with live YOLO feedback.
            pan_angle = track_person_closed_loop(
                picam2,
                vision_model,
                pan_motion,
                initial_error_sign,
                confirmed_cx,
            )


            print(f"Tracking movement complete: {pan_angle:.1f} deg")


            time.sleep(POST_MOVE_REACQUIRE_DELAY)




    except Exception as e:
        print()
        print("VISION ERROR:", e)
        print("Voice assistant remains online; vision will be restarted.")


    finally:
        if pan_motion is not None:
            try:
                pan_motion.shutdown()
            except Exception:
                pass
        elif pan_servo is not None:
            try:
                pan_servo.stop()
            except Exception:
                pass


        if picam2 is not None:
            try:
                if locals().get("camera_running", False):
                    picam2.stop()
            except Exception:
                pass


        print("GLaDOS vision tracker stopped.")


def _normalize_voice_command(user_input):
    return " ".join(
        user_input.lower().strip().replace("-", " " ).split()
    )




def is_sleep_command(user_input):
    """Return True for commands that should place GLaDOS in standby."""
    normalized = _normalize_voice_command(user_input)


    return normalized in {
        "sleep",
        "go to sleep",
        "go sleep",
        "sleep glados",
        "glados sleep",
        "standby",
        "stand by",
        "enter standby",
        "go into standby",
        "rest",
        "go rest",
        "go offline",
        # Keep the old natural shutdown phrases safe/wakeable too.
        "shutdown",
        "shut down",
        "shutdown glados",
        "shut down glados",
        "glados shutdown",
        "glados shut down",
        "power down",
        "power down glados",
    }




def is_terminate_command(user_input):
    """Return True only for deliberately explicit full program exits."""
    normalized = _normalize_voice_command(user_input)


    return normalized in {
        "terminate program",
        "terminate glados",
        "full shutdown",
        "full shut down",
        "fully shut down",
        "shutdown completely",
        "shut down completely",
        "exit program",
        "exit glados",
        "quit program",
        "stop program",
    }




def assistant_loop():
    while not shutdown_event.is_set():


       try:


           #waits for the wakeword to be spoken, then listens in
           with SuppressStderr():
               user_input = listen_for_wake_and_command()


           if user_input is None:
               assistant_busy_event.clear()
               print("Returning to wake word mode...")
               continue


           # --------------------------------------------------------
           # LOCAL SLEEP / STANDBY
           # --------------------------------------------------------
           # Everyday "sleep" and old "shutdown" phrases no longer kill
           # Python. They put GLaDOS into a quiet, wake-word-only standby.
           if is_sleep_command(user_input):
               print("Voice sleep command received.")


               speak("Entering standby.")


               standby_event.set()
               assistant_busy_event.clear()
               print("GLaDOS is sleeping. Say Hey GLaDOS to wake her.")
               continue


           # --------------------------------------------------------
           # TRUE PROGRAM TERMINATION
           # --------------------------------------------------------
           # Requires an intentionally explicit phrase so normal sleep cannot
           # accidentally make GLaDOS unreachable by voice.
           if is_terminate_command(user_input):
               print("Voice program termination command received.")


               speak("Shutting down completely.")


               assistant_busy_event.clear()
               shutdown_event.set()
               break




           # Reminder actions are checked first so phrases such as
           # "forget my reminder to..." delete from the reminders table
           # instead of being misrouted to the generic memory table.
           if should_check_reminder(user_input):
               if process_reminder(user_input):
                   assistant_busy_event.clear()
                   continue


           # Only run the extra Gemini classifiers when the wording suggests
           # that feature is actually being requested. Normal conversation
           # therefore goes directly to GLaDOS with one Gemini request.
           if should_check_memory_save(user_input):
               process_memory(user_input)


           if should_check_forget(user_input):
               if process_forget(user_input):
                   assistant_busy_event.clear()
                   continue


           memory_context = get_memory_context()


           prompt = f"""
       Here are the memories stored for the user:


       {memory_context}


       The user said:
       "{user_input}"


       Use the stored memories only if they are relevant to the current message.
       Otherwise ignore them.
       """




           try:
               gemini_start = time.monotonic()


               response = gemini_with_retry(
                   lambda: conversation.send_message(prompt),
                   label="GLaDOS response",
               )


               if PRINT_LATENCY_TIMINGS:
                   print(
                       f"Latency | Gemini response: "
                       f"{time.monotonic() - gemini_start:.2f}s"
                   )


           except Exception as e:
               print("GLaDOS response failed after retries:", e)


               # Piper is local, so GLaDOS can still acknowledge the problem
               # even when the cloud model is temporarily unavailable.
               # Thinking is over. Resume YOLO while GLaDOS speaks.
               assistant_busy_event.clear()


               speak(
                   "Apparently the external network has decided to become "
                   "temporarily useless. Try again in a moment."
               )
               continue


           print(
               "GLaDOS:",
               response.text
           )


           # Resume YOLO before speech starts so GLaDOS can track while talking.
           assistant_busy_event.clear()


           speak(
               response.text
           )


           # Then the loop automatically goes back
           # to requiring "Hey GLaDOS" again.


       except sr.UnknownValueError:


           assistant_busy_event.clear()


           print(
               "Couldn't understand command."
           )


       except sr.RequestError as e:


           assistant_busy_event.clear()


           print(
               "Speech recognition error:",
               e
           )


       except KeyboardInterrupt:


           assistant_busy_event.clear()


           print(
               "Stopping GLaDOS..."
           )
           shutdown_event.set()
           break


       except Exception as e:


           assistant_busy_event.clear()


           print(
               "Voice loop error; recovering:",
               e
           )


           # Give ALSA/PortAudio a moment to release/re-enumerate the device
           # before opening a fresh stream on the next wake-listener cycle.
           shutdown_event.wait(VOICE_LOOP_RECOVERY_DELAY)




def run_glados():
    reminder_thread = threading.Thread(
        target=reminder_loop,
        daemon=True,
    )
    reminder_thread.start()


    # Voice waits for "Hey GLaDOS" in parallel with the vision tracker.
    assistant_thread = threading.Thread(
        target=assistant_loop,
        daemon=True,
    )
    assistant_thread.start()


    try:
        # Keep PiCamera/YOLO on the main thread, but recover it independently
        # from the voice assistant. A camera/YOLO hiccup must never make
        # GLaDOS stop listening for the wake word.
        while not shutdown_event.is_set():
            vision_tracking_loop()


            if not shutdown_event.is_set():
                print(
                    f"Restarting vision system in {VISION_RETRY_DELAY:.1f} seconds..."
                )
                shutdown_event.wait(VISION_RETRY_DELAY)


    except KeyboardInterrupt:
        print()
        print("Stopping GLaDOS...")
        shutdown_event.set()
    finally:
        shutdown_event.set()




if __name__ == "__main__":
    run_glados()




