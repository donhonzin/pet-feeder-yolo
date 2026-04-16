#!/usr/bin/env python3

import os
import json
import time
import cv2
import threading
from collections import deque

from picamera2 import Picamera2
from ultralytics import YOLO
import RPi.GPIO as GPIO


class FeederCore:
    def __init__(
        self,
        model_path="/home/jankriz/petfeeder/models/best_ncnn_model",
        config_path="/home/jankriz/petfeeder/pet_config.json",
        history_path="/home/jankriz/petfeeder/feeding_history.json",
        servo_pin=17,
        pwm_frequency=50,
        stop_dc=7.5,
        forward_dc=9.5,
        live_width=640,
        live_height=480,
        detect_width=320,
        detect_height=320,
        rotate_frame=True,
        detection_interval=0.6,
        jpeg_quality=45,
        manual_feed_seconds=2.0,
        preview_fps_limit=15,
    ):
        self.model_path = model_path
        self.config_path = config_path
        self.history_path = history_path

        self.servo_pin = servo_pin
        self.pwm_frequency = pwm_frequency
        self.stop_dc = stop_dc
        self.forward_dc = forward_dc

        self.live_width = live_width
        self.live_height = live_height
        self.detect_width = detect_width
        self.detect_height = detect_height
        self.rotate_frame = rotate_frame
        self.detection_interval = detection_interval
        self.jpeg_quality = jpeg_quality
        self.manual_feed_seconds = manual_feed_seconds
        self.preview_fps_limit = preview_fps_limit

        self.config_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.feed_lock = threading.Lock()
        self.history_lock = threading.Lock()
        self.state_lock = threading.Lock()

        self.pet_config = {}
        self.config_last_modified = 0
        self.last_feed_times = {}

        self.latest_jpeg = None
        self.latest_detect_frame = None

        self.latest_status = "Starting..."
        self.latest_detection = "No detection yet"

        self.shutdown_flag = False
        self.threads_started = False

        self.picam2 = None
        self.model = None
        self.class_names = None

        self.history_cache = deque(maxlen=100)

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.servo_pin, GPIO.OUT)
        GPIO.output(self.servo_pin, GPIO.LOW)

    # =====================================================
    # FILE / CONFIG / HISTORY
    # =====================================================

    def ensure_history_file(self):
        if not os.path.exists(self.history_path):
            with open(self.history_path, "w") as f:
                json.dump([], f, indent=4)

    def load_history(self):
        self.ensure_history_file()
        with self.history_lock:
            with open(self.history_path, "r") as f:
                try:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
                except json.JSONDecodeError:
                    return []

    def save_history(self, entries):
        with self.history_lock:
            with open(self.history_path, "w") as f:
                json.dump(entries, f, indent=4)

    def append_history_entry(self, pet_id, portion, method):
        entries = self.load_history()
        entry = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pet_id": pet_id,
            "portion": portion,
            "method": method,
        }
        entries.append(entry)

        if len(entries) > 200:
            entries = entries[-200:]

        self.save_history(entries)
        self.history_cache.clear()
        self.history_cache.extend(entries[-100:])

    def get_last_fed_map(self):
        entries = self.load_history()
        last_fed = {}

        for entry in entries:
            pet_id = entry.get("pet_id")
            if pet_id and pet_id != "Manual":
                last_fed[pet_id] = entry.get("time", "Unknown")

        return last_fed

    def get_last_manual_feed_time(self):
        entries = self.load_history()
        for entry in reversed(entries):
            if entry.get("method") == "manual":
                return entry.get("time", "Unknown")
        return "Never"

    def get_recent_activity(self, limit=5):
        entries = self.load_history()
        return list(reversed(entries[-limit:]))

    def load_pet_config(self):
        with open(self.config_path, "r") as f:
            return json.load(f)

    def save_pet_config(self, config):
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=4)

    def sync_last_feed_times(self):
        for pet_name in self.pet_config:
            if pet_name not in self.last_feed_times:
                self.last_feed_times[pet_name] = 0

    def initialize_config(self):
        with self.config_lock:
            self.pet_config = self.load_pet_config()
            self.config_last_modified = os.path.getmtime(self.config_path)
            self.sync_last_feed_times()

    def reload_config_if_changed(self):
        try:
            current_modified = os.path.getmtime(self.config_path)
        except FileNotFoundError:
            return

        if current_modified != self.config_last_modified:
            with self.config_lock:
                self.pet_config = self.load_pet_config()
                self.config_last_modified = current_modified
                self.sync_last_feed_times()

    # =====================================================
    # STATE HELPERS
    # =====================================================

    def set_status(self, status=None, detection=None):
        with self.state_lock:
            if status is not None:
                self.latest_status = status
            if detection is not None:
                self.latest_detection = detection

    def get_status(self):
        with self.state_lock:
            return {
                "status": self.latest_status,
                "detection": self.latest_detection,
            }

    # =====================================================
    # FEEDING / GPIO
    # =====================================================

    def dispense(self, seconds, pet_name="Manual", method="manual"):
        with self.feed_lock:
            pwm = GPIO.PWM(self.servo_pin, self.pwm_frequency)

            try:
                self.set_status(status=f"Dispensing for {seconds:.1f}s")
                pwm.start(0)
                time.sleep(0.05)

                pwm.ChangeDutyCycle(self.forward_dc)
                time.sleep(seconds)

                pwm.ChangeDutyCycle(self.stop_dc)
                time.sleep(0.10)

                pwm.ChangeDutyCycle(0)
                time.sleep(0.05)

                self.append_history_entry(
                    pet_id=pet_name,
                    portion=f"{seconds:.1f}s",
                    method=method,
                )

            finally:
                pwm.stop()
                GPIO.output(self.servo_pin, GPIO.LOW)
                self.set_status(status="Idle")

    def manual_feed(self):
        self.dispense(
            seconds=self.manual_feed_seconds,
            pet_name="Manual",
            method="manual",
        )

    def can_feed_pet(self, pet_name, now):
        with self.config_lock:
            if pet_name not in self.pet_config:
                return False, 0
            cooldown = self.pet_config[pet_name]["cooldown_seconds"]

        last_feed = self.last_feed_times.get(pet_name, 0)
        elapsed = now - last_feed

        if elapsed >= cooldown:
            return True, 0

        return False, cooldown - elapsed

    # =====================================================
    # MODEL / DETECTION
    # =====================================================

    def get_best_detection(self, results):
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None, None

        confs = boxes.conf.tolist()
        class_ids = boxes.cls.tolist()

        best_index = max(range(len(confs)), key=lambda i: confs[i])
        best_conf = confs[best_index]
        best_class_id = int(class_ids[best_index])

        return self.class_names[best_class_id], best_conf

    # =====================================================
    # CAMERA / WORKERS
    # =====================================================

    def _rotate_if_needed(self, frame):
        if self.rotate_frame:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame

    def camera_worker(self):
        min_frame_time = 1.0 / max(1, self.preview_fps_limit)
        last_jpeg_time = 0.0

        while not self.shutdown_flag:
            try:
                (main_frame, lores_frame), _ = self.picam2.capture_arrays(["main", "lores"])

                # main stream is RGB888
                main_frame = self._rotate_if_needed(main_frame)

                # lores stream is YUV420, convert once for YOLO
                detect_bgr = cv2.cvtColor(lores_frame, cv2.COLOR_YUV2BGR_I420)
                detect_bgr = self._rotate_if_needed(detect_bgr)

                now = time.time()

                with self.frame_lock:
                    self.latest_detect_frame = detect_bgr

                # Only JPEG-encode often enough for a smooth preview, not max speed
                if now - last_jpeg_time >= min_frame_time:
                    ok, jpeg = cv2.imencode(
                        ".jpg",
                        main_frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                    )
                    if ok:
                        with self.frame_lock:
                            self.latest_jpeg = jpeg.tobytes()
                    last_jpeg_time = now

            except Exception as e:
                print("camera_worker error:", e)
                time.sleep(0.05)

    def detection_worker(self):
        last_detect_time = 0.0

        while not self.shutdown_flag:
            try:
                self.reload_config_if_changed()

                now = time.time()
                if now - last_detect_time < self.detection_interval:
                    time.sleep(0.02)
                    continue

                with self.frame_lock:
                    frame = self.latest_detect_frame

                if frame is None:
                    time.sleep(0.05)
                    continue

                # YOLO predict mode is the standard inference path in Ultralytics. :contentReference[oaicite:1]{index=1}
                results = self.model.predict(frame, verbose=False)
                pet_name, confidence = self.get_best_detection(results)

                detect_time = time.time()

                if pet_name is not None:
                    with self.config_lock:
                        pet_settings = self.pet_config.get(pet_name, {})
                        dispense_seconds = pet_settings.get("dispense_seconds", 2.0)
                        threshold = pet_settings.get("confidence_threshold", 0.60)

                    if confidence < threshold:
                        self.set_status(
                            status="Idle",
                            detection=f"{pet_name} low confidence ({confidence:.2f})",
                        )
                    else:
                        allowed, remaining = self.can_feed_pet(pet_name, detect_time)

                        if allowed:
                            self.set_status(
                                status=f"Auto feed: {pet_name}",
                                detection=f"{pet_name} detected ({confidence:.2f})",
                            )
                            self.last_feed_times[pet_name] = detect_time

                            threading.Thread(
                                target=self.dispense,
                                kwargs={
                                    "seconds": dispense_seconds,
                                    "pet_name": pet_name,
                                    "method": "auto",
                                },
                                daemon=True,
                            ).start()
                        else:
                            self.set_status(
                                status="Cooldown active",
                                detection=(
                                    f"{pet_name} detected ({confidence:.2f}) - "
                                    f"{remaining/60:.1f} min left"
                                ),
                            )
                else:
                    self.set_status(status="Idle", detection="No valid detection")

                last_detect_time = detect_time

            except Exception as e:
                print("detection_worker error:", e)
                self.set_status(status=f"Error: {e}")
                time.sleep(0.5)

    # =====================================================
    # PUBLIC API
    # =====================================================

    def start(self):
        self.ensure_history_file()
        self.initialize_config()

        self.picam2 = Picamera2()

        # Picamera2 supports main + lores streams in one configuration. :contentReference[oaicite:2]{index=2}
        config = self.picam2.create_video_configuration(
            main={"size": (self.live_width, self.live_height), "format": "RGB888"},
            lores={"size": (self.detect_width, self.detect_height), "format": "YUV420"},
            buffer_count=4,
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1.5)

        self.model = YOLO(self.model_path, task="detect")
        self.class_names = self.model.names

        if not self.threads_started:
            threading.Thread(target=self.camera_worker, daemon=True).start()
            threading.Thread(target=self.detection_worker, daemon=True).start()
            self.threads_started = True

        self.set_status(status="Idle", detection="Ready")

    def stop(self):
        self.shutdown_flag = True
        try:
            if self.picam2 is not None:
                self.picam2.stop()
        except Exception:
            pass

        GPIO.output(self.servo_pin, GPIO.LOW)
        GPIO.cleanup()

    def get_latest_jpeg(self):
        with self.frame_lock:
            return self.latest_jpeg

    def get_pet_config(self):
        with self.config_lock:
            return dict(self.pet_config)

    def update_pet_config(self, new_config):
        with self.config_lock:
            self.save_pet_config(new_config)
            self.pet_config = new_config
            self.config_last_modified = os.path.getmtime(self.config_path)
            self.sync_last_feed_times()
