#!/usr/bin/env python3

import time
import json
import os
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import RPi.GPIO as GPIO

# =========================================================
# CONFIG
# =========================================================

# Servo pin:
# - physical pin 11 -> BCM 17
# - physical pin 12 -> BCM 18
SERVO_PIN = 17

PWM_FREQUENCY = 50

# Continuous servo values
STOP_DC = 7.5
FORWARD_DC = 9.5

# Camera / model
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
ROTATE_FRAME = True
MODEL_PATH = "/home/jankriz/petfeeder/models/best_ncnn_model"
CONFIG_PATH = "/home/jankriz/petfeeder/pet_config.json"

# Loop timing
LOOP_DELAY_SECONDS = 1.0

DEBUG = True


# =========================================================
# GPIO SETUP
# =========================================================

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.output(SERVO_PIN, GPIO.LOW)


# =========================================================
# SERVO CONTROL
# =========================================================

def dispense(seconds):
    """
    Spin the continuous servo for a short time, then stop fully.
    Keeps the servo quiet while idle.
    """
    pwm = GPIO.PWM(SERVO_PIN, PWM_FREQUENCY)

    try:
        pwm.start(0)
        time.sleep(0.05)

        # Run feeder
        pwm.ChangeDutyCycle(FORWARD_DC)
        time.sleep(seconds)

        # Brief neutral pulse to settle
        pwm.ChangeDutyCycle(STOP_DC)
        time.sleep(0.10)

        # No signal -> quieter idle
        pwm.ChangeDutyCycle(0)
        time.sleep(0.05)

    finally:
        pwm.stop()
        GPIO.output(SERVO_PIN, GPIO.LOW)


# =========================================================
# CONFIG LOADING (HOT RELOAD)
# =========================================================

def load_pet_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


PET_CONFIG = load_pet_config()
config_last_modified = os.path.getmtime(CONFIG_PATH)


def reload_config_if_changed():
    global PET_CONFIG, config_last_modified

    current_modified = os.path.getmtime(CONFIG_PATH)

    if current_modified != config_last_modified:
        PET_CONFIG = load_pet_config()
        config_last_modified = current_modified
        print("Config reloaded")


# =========================================================
# CAMERA SETUP
# =========================================================

picam2 = Picamera2()
picam2.preview_configuration.main.size = (FRAME_WIDTH, FRAME_HEIGHT)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
time.sleep(2)


# =========================================================
# MODEL SETUP
# =========================================================

model = YOLO(MODEL_PATH, task="detect")
class_names = model.names


# =========================================================
# FEED TRACKING
# =========================================================

# Last successful feed time per pet
last_feed_times = {
    pet_name: 0 for pet_name in PET_CONFIG
}


# =========================================================
# HELPERS
# =========================================================

def get_best_detection(results):
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None, None

    confs = boxes.conf.tolist()
    class_ids = boxes.cls.tolist()

    best_index = max(range(len(confs)), key=lambda i: confs[i])
    best_conf = confs[best_index]
    best_class_id = int(class_ids[best_index])

    pet_name = class_names[best_class_id]
    return pet_name, best_conf


def can_feed_pet(pet_name, now):
    if pet_name not in PET_CONFIG:
        return False, 0

    cooldown = PET_CONFIG[pet_name]["cooldown_seconds"]
    last_feed = last_feed_times.get(pet_name, 0)
    elapsed = now - last_feed

    if elapsed >= cooldown:
        return True, 0

    remaining = cooldown - elapsed
    return False, remaining


# =========================================================
# MAIN LOOP
# =========================================================

try:
    while True:
        reload_config_if_changed()

        frame = picam2.capture_array()

        if ROTATE_FRAME:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        results = model.predict(frame, verbose=False)
        pet_name, confidence = get_best_detection(results)

        now = time.time()

        if pet_name is not None:
            pet_settings = PET_CONFIG.get(pet_name, {})

            dispense_seconds = pet_settings.get("dispense_seconds", 2.0)
            threshold = pet_settings.get("confidence_threshold", 0.60)

            if confidence < threshold:
                if DEBUG:
                    print(f"{pet_name} detected but low confidence ({confidence:.2f} < {threshold:.2f})")
            else:
                allowed, remaining = can_feed_pet(pet_name, now)

                if allowed:
                    print(
                        f"Detected: {pet_name} "
                        f"(conf={confidence:.2f}) -> dispensing for {dispense_seconds:.1f}s"
                    )
                    dispense(dispense_seconds)
                    last_feed_times[pet_name] = time.time()
                else:
                    if DEBUG:
                        remaining_mins = remaining / 60
                        print(
                            f"Detected: {pet_name} "
                            f"(conf={confidence:.2f}) but cooldown active "
                            f"({remaining_mins:.1f} min left)"
                        )

        time.sleep(LOOP_DELAY_SECONDS)

finally:
    picam2.stop()
    GPIO.output(SERVO_PIN, GPIO.LOW)
    GPIO.cleanup()
    print("Clean shutdown ✅")
