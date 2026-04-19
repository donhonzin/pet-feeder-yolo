#!/usr/bin/env python3

import threading
import json
from functools import wraps

import cv2
import numpy as np

from flask import (
    Flask,
    Response,
    redirect,
    render_template_string,
    request,
    url_for,
    session,
)
from werkzeug.security import check_password_hash

from feeder_core import FeederCore

app = Flask(__name__)
app.secret_key = "petfeeder-secret-8392"
app.config["SESSION_COOKIE_SECURE"] = True
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

core = FeederCore(
    model_path="/home/jankriz/petfeeder/models/best_ncnn_model",
    config_path="/home/jankriz/petfeeder/pet_config.json",
    history_path="/home/jankriz/petfeeder/feeding_history.json",
    servo_pin=17,
    live_width=960,
    live_height=540,
    detect_width=320,
    detect_height=320,
    rotate_frame=True,
    detection_interval=1.0,
    jpeg_quality=45,
    manual_feed_seconds=2.0,
    preview_fps_limit=24,
)

AUTH_CONFIG_PATH = "/home/jankriz/petfeeder/auth_config.json"

COOLDOWN_OPTIONS = [
    (15, "15 min"), (30, "30 min"), (45, "45 min"), (60, "1 hr"),
    (75, "1 hr 15 min"), (90, "1 hr 30 min"), (105, "1 hr 45 min"),
    (120, "2 hrs"), (135, "2 hrs 15 min"), (150, "2 hrs 30 min"),
    (165, "2 hrs 45 min"), (180, "3 hrs"), (195, "3 hrs 15 min"),
    (210, "3 hrs 30 min"), (225, "3 hrs 45 min"), (240, "4 hrs"),
    (255, "4 hrs 15 min"), (270, "4 hrs 30 min"), (285, "4 hrs 45 min"),
    (300, "5 hrs"), (315, "5 hrs 15 min"), (330, "5 hrs 30 min"),
    (345, "5 hrs 45 min"), (360, "6 hrs"),
]

CONFIDENCE_OPTIONS = [
    (0.60, "0.60"), (0.65, "0.65"), (0.70, "0.70"),
    (0.75, "0.75"), (0.80, "0.80"), (0.85, "0.85"), (0.90, "0.90"),
]

DISPENSE_OPTIONS = [(i, f"{i} sec") for i in range(0, 9)]


def load_auth_config():
    with open(AUTH_CONFIG_PATH, "r") as f:
        return json.load(f)


def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login", next=request.path))
        return func(*args, **kwargs)
    return wrapper


def get_color_for_label(label):
    label_key = str(label).lower().strip()

    palette = {
        "itchy": (0, 255, 255),   # yellow
        "nuke": (255, 0, 0),      # blue
        "lily": (0, 0, 255),      # red
    }

    if label_key in palette:
        return palette[label_key]

    seed = sum(ord(c) for c in label_key)
    return (
        50 + (seed * 53) % 206,
        50 + (seed * 97) % 206,
        50 + (seed * 193) % 206,
    )


def get_latest_detections():
    """
    Tries to obtain live detections from FeederCore.
    Supported formats:
      - core.get_latest_detections() -> list[dict]
      - core.latest_detections -> list[dict]

    Expected dict keys (flexible):
      x1, y1, x2, y2
      OR bbox = [x1, y1, x2, y2]
      label / pet_id / class_name / name
      confidence / conf / score
    """
    try:
        if hasattr(core, "get_latest_detections") and callable(core.get_latest_detections):
            detections = core.get_latest_detections()
            if isinstance(detections, list):
                return detections
    except Exception:
        pass

    try:
        detections = getattr(core, "latest_detections", None)
        if isinstance(detections, list):
            return detections
    except Exception:
        pass

    return []


def normalize_detection(det):
    if not isinstance(det, dict):
        return None

    if "bbox" in det and isinstance(det["bbox"], (list, tuple)) and len(det["bbox"]) == 4:
        x1, y1, x2, y2 = det["bbox"]
    else:
        required = ["x1", "y1", "x2", "y2"]
        if not all(k in det for k in required):
            return None
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]

    label = (
        det.get("label")
        or det.get("pet_id")
        or det.get("class_name")
        or det.get("name")
        or "Pet"
    )

    confidence = det.get("confidence", det.get("conf", det.get("score", 0.0)))

    try:
        x1 = int(float(x1))
        y1 = int(float(y1))
        x2 = int(float(x2))
        y2 = int(float(y2))
        confidence = float(confidence)
    except Exception:
        return None

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "label": str(label),
        "confidence": confidence,
    }


def draw_detection_overlays(frame):
    detections = get_latest_detections()
    if not detections:
        return frame

    h, w = frame.shape[:2]

    for raw_det in detections:
        det = normalize_detection(raw_det)
        if not det:
            continue

        x1 = max(0, min(w - 1, det["x1"]))
        y1 = max(0, min(h - 1, det["y1"]))
        x2 = max(0, min(w - 1, det["x2"]))
        y2 = max(0, min(h - 1, det["y2"]))

        if x2 <= x1 or y2 <= y1:
            continue

        label = det["label"]
        confidence = det["confidence"]
        color = get_color_for_label(label)

        text = f"{label} {confidence * 100:.1f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        text_bg_y1 = max(0, y1 - text_h - baseline - 10)
        text_bg_y2 = max(text_h + baseline + 8, y1)

        cv2.rectangle(
            frame,
            (x1, text_bg_y1),
            (min(w - 1, x1 + text_w + 12), text_bg_y2),
            color,
            -1,
        )

        cv2.putText(
            frame,
            text,
            (x1 + 6, text_bg_y2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return frame


def annotate_jpeg_frame(frame_bytes):
    np_frame = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

    if frame is None:
        return frame_bytes

    frame = draw_detection_overlays(frame)

    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(core.jpeg_quality)],
    )
    if not success:
        return frame_bytes

    return encoded.tobytes()


LOGIN_HTML = """
<!doctype html>
<html>
<head>
    <title>Pet Feeder Login</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            margin: 0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #060b06;
            color: #e8ef3d;
            font-family: Arial, sans-serif;
        }
        .card {
            width: 100%;
            max-width: 380px;
            border: 2px solid #d7df00;
            border-radius: 14px;
            padding: 24px;
            background: #0d150d;
            box-shadow: 0 0 18px rgba(215, 223, 0, 0.15);
        }
        h1 {
            margin-top: 0;
            text-align: center;
        }
        label {
            display: block;
            margin-top: 14px;
            margin-bottom: 6px;
            font-weight: bold;
        }
        input {
            width: 100%;
            padding: 10px 12px;
            border-radius: 8px;
            border: 2px solid #d7df00;
            background: #060b06;
            color: #e8ef3d;
            box-sizing: border-box;
            font-size: 16px;
        }
        button {
            width: 100%;
            margin-top: 18px;
            padding: 12px;
            font-size: 16px;
            font-weight: bold;
            border-radius: 10px;
            border: 2px solid #d7df00;
            background: #d7df00;
            color: #000;
            cursor: pointer;
        }
        .error {
            margin-top: 12px;
            color: #ff8080;
            text-align: center;
        }
        .small {
            margin-top: 10px;
            color: #aab200;
            text-align: center;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>Pet Feeder Login</h1>
        <form method="post">
            <label>Username</label>
            <input type="text" name="username" required>

            <label>Password</label>
            <input type="password" name="password" required>

            <button type="submit">Log In</button>
        </form>

        {% if error %}
            <div class="error">{{ error }}</div>
        {% endif %}

        <div class="small">Dashboard access is protected.</div>
    </div>
</body>
</html>
"""


BASE_HTML = """
<!doctype html>
<html>
<head>
    <title>Smart IoT Pet Feeder</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        :root {
            --bg: #060b06;
            --line: #d7df00;
            --text: #e8ef3d;
            --muted: #aab200;
            --btn: #0d150d;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
        }
        .layout {
            display: flex;
            min-height: 100vh;
        }
        .sidebar {
            width: 180px;
            border-right: 2px solid var(--line);
            padding: 12px 10px;
        }
        .nav-btn {
            display: block;
            width: 100%;
            text-decoration: none;
            color: var(--text);
            border: 2px solid var(--line);
            background: var(--btn);
            padding: 12px 10px;
            margin-bottom: 10px;
            text-align: center;
            font-weight: bold;
            border-radius: 10px;
            transition: transform 0.18s ease, background 0.18s ease, color 0.18s ease, box-shadow 0.18s ease;
        }
        .nav-btn:hover {
            transform: translateX(4px);
            box-shadow: 0 0 12px rgba(215, 223, 0, 0.35);
        }
        .nav-btn.active {
            background: #d7df00;
            color: #000;
            box-shadow: 0 0 14px rgba(215, 223, 0, 0.45);
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border: 2px solid var(--line);
            border-radius: 999px;
            background: var(--btn);
            font-weight: bold;
            margin-bottom: 14px;
        }
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #39ff14;
            box-shadow: 0 0 8px rgba(57, 255, 20, 0.7);
            animation: pulse 1.6s infinite ease-in-out;
        }
        .status-dot.warning {
            background: #ffd000;
            box-shadow: 0 0 8px rgba(255, 208, 0, 0.7);
        }
        .status-dot.error {
            background: #ff4d4d;
            box-shadow: 0 0 8px rgba(255, 77, 77, 0.7);
        }
        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.18); opacity: 0.75; }
            100% { transform: scale(1); opacity: 1; }
        }
        .content {
            flex: 1;
            padding: 18px;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 28px;
        }
        .panel {
            border: 2px solid var(--line);
            padding: 14px;
            min-height: 260px;
        }
        .panel h2 {
            text-align: center;
            margin-top: 0;
            margin-bottom: 14px;
            font-size: 26px;
        }
        .feed-box {
            width: 100%;
            min-height: 300px;
            border: 2px solid var(--line);
            background: #000;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .feed-box img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }
        .manual-card {
            min-height: 300px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 18px;
            text-align: center;
        }
        .manual-subtitle {
            color: var(--muted);
            font-size: 16px;
        }
        .manual-meta {
            width: 100%;
            max-width: 420px;
            display: grid;
            gap: 12px;
            margin-top: 8px;
        }
        .manual-meta-item {
            border: 2px solid var(--line);
            border-radius: 10px;
            background: var(--btn);
            padding: 12px;
        }
        .action-btn {
            padding: 18px 28px;
            font-size: 28px;
            border: 2px solid var(--line);
            background: var(--btn);
            color: var(--text);
            border-radius: 10px;
            cursor: pointer;
            font-weight: bold;
        }
        .action-btn:hover {
            box-shadow: 0 0 12px rgba(215, 223, 0, 0.35);
        }
        .pet-actions {
            max-width: 340px;
            margin: 0 auto;
        }
        .pet-actions a {
            display: block;
            text-decoration: none;
            color: var(--text);
            border: 2px solid var(--line);
            border-radius: 10px;
            background: var(--btn);
            padding: 14px;
            margin-bottom: 12px;
            text-align: center;
            font-weight: bold;
            font-size: 18px;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        .pet-actions a:hover {
            transform: translateY(-2px);
            box-shadow: 0 0 12px rgba(215, 223, 0, 0.35);
        }
        .last-fed-list {
            margin-bottom: 18px;
        }
        .last-fed-item {
            border: 2px solid var(--line);
            border-radius: 10px;
            padding: 10px 12px;
            margin-bottom: 10px;
            background: var(--btn);
        }
        .last-fed-name {
            font-weight: bold;
            font-size: 18px;
            margin-bottom: 4px;
        }
        .activity-list {
            display: grid;
            gap: 12px;
        }
        .activity-card {
            border: 2px solid var(--line);
            border-radius: 10px;
            background: var(--btn);
            padding: 12px;
        }
        .activity-top {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
            margin-bottom: 6px;
        }
        .activity-pet {
            font-weight: bold;
            font-size: 18px;
        }
        .activity-time {
            color: var(--muted);
            font-size: 14px;
        }
        .activity-meta {
            color: var(--text);
            font-size: 15px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            color: var(--text);
        }
        th, td {
            border: 2px solid var(--line);
            padding: 10px;
            text-align: center;
        }
        .status-bar {
            margin-bottom: 18px;
            border: 2px solid var(--line);
            padding: 10px 14px;
            display: flex;
            justify-content: space-between;
            gap: 20px;
            flex-wrap: wrap;
        }
        .form-grid {
            display: grid;
            gap: 18px;
        }
        .pet-card {
            border: 2px solid var(--line);
            padding: 16px;
        }
        .pet-meta {
            margin-bottom: 14px;
            color: var(--muted);
            font-size: 14px;
        }
        label {
            display: block;
            margin-bottom: 6px;
            font-weight: bold;
        }
        select {
            width: 100%;
            max-width: 320px;
            padding: 10px;
            background: var(--btn);
            color: var(--text);
            border: 2px solid var(--line);
            border-radius: 8px;
        }
        .save-row { margin-top: 18px; }
        .logout-row {
            margin-top: 18px;
        }
        .logout-btn {
            display: block;
            width: 100%;
            text-decoration: none;
            text-align: center;
            border: 2px solid var(--line);
            border-radius: 10px;
            background: var(--btn);
            color: var(--text);
            padding: 10px;
            font-weight: bold;
        }
        .small-note {
            margin-top: 8px;
            color: var(--muted);
            font-size: 14px;
        }
        @media (max-width: 900px) {
            .layout { flex-direction: column; }
            .sidebar {
                width: 100%;
                border-right: none;
                border-bottom: 2px solid var(--line);
            }
            .grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
<div class="layout">
    <aside class="sidebar">
        <div class="status-pill">
            <span class="status-dot {% if status_class != 'ok' %}{{ status_class }}{% endif %}"></span>
            <span>{{ status_label }}</span>
        </div>

        <a class="nav-btn {% if request.endpoint == 'index' %}active{% endif %}" href="{{ url_for('index') }}">HOME</a>
        <a class="nav-btn {% if request.endpoint == 'live_feed_page' %}active{% endif %}" href="{{ url_for('live_feed_page') }}">LIVE FEED</a>
        <a class="nav-btn {% if request.endpoint == 'manual_page' %}active{% endif %}" href="{{ url_for('manual_page') }}">MANUAL FEEDING</a>
        <a class="nav-btn {% if request.endpoint == 'pets_page' %}active{% endif %}" href="{{ url_for('pets_page') }}">PET PROFILES</a>
        <a class="nav-btn {% if request.endpoint == 'history_page' %}active{% endif %}" href="{{ url_for('history_page') }}">FEEDING HISTORY</a>

        <div class="logout-row">
            <a class="logout-btn" href="{{ url_for('logout') }}">LOG OUT</a>
        </div>
    </aside>
    <main class="content">
        <div class="status-bar">
            <div><strong>Status:</strong> {{ status }}</div>
            <div><strong>Detection:</strong> {{ detection }}</div>
        </div>
        {{ body|safe }}
    </main>
</div>
</body>
</html>
"""


def render_page(body):
    state = core.get_status()
    status_text = state["status"].lower()

    if "error" in status_text:
        status_class = "error"
        status_label = "System Error"
    elif "cooldown" in status_text:
        status_class = "warning"
        status_label = "Cooldown Active"
    elif "auto feed" in status_text or "dispensing" in status_text:
        status_class = "warning"
        status_label = "Feeding"
    else:
        status_class = "ok"
        status_label = "System Online"

    return render_template_string(
        BASE_HTML,
        body=body,
        status=state["status"],
        detection=state["detection"],
        status_class=status_class,
        status_label=status_label,
    )


def render_history_table(limit=10):
    rows = core.get_recent_activity(limit)
    table_html = """
    <table>
        <thead>
            <tr>
                <th>TIME</th>
                <th>PET ID</th>
                <th>PORTION</th>
                <th>METHOD</th>
            </tr>
        </thead>
        <tbody>
    """
    for row in rows:
        table_html += f"""
            <tr>
                <td>{row['time']}</td>
                <td>{row['pet_id']}</td>
                <td>{row['portion']}</td>
                <td>{row['method']}</td>
            </tr>
        """
    if not rows:
        table_html += '<tr><td colspan="4">No history yet</td></tr>'
    table_html += "</tbody></table>"
    return table_html


def render_recent_activity(limit=5):
    rows = core.get_recent_activity(limit)
    if not rows:
        return """
        <div class="activity-card">
            <div class="activity-meta">No recent feeding activity</div>
        </div>
        """

    html = '<div class="activity-list">'
    for row in rows:
        html += f"""
        <div class="activity-card">
            <div class="activity-top">
                <div class="activity-pet">{row['pet_id']}</div>
                <div class="activity-time">{row['time']}</div>
            </div>
            <div class="activity-meta">
                Portion: {row['portion']} &nbsp;•&nbsp; Method: {row['method']}
            </div>
        </div>
        """
    html += "</div>"
    return html


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("index"))

    error = None
    auth = load_auth_config()

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        next_url = request.args.get("next") or url_for("index")

        if username == auth["username"] and check_password_hash(auth["password_hash"], password):
            session.clear()
            session["logged_in"] = True
            session["username"] = username
            return redirect(next_url)

        error = "Invalid username or password."

    return render_template_string(LOGIN_HTML, error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    last_fed = core.get_last_fed_map()
    manual_last = core.get_last_manual_feed_time()
    pet_config = core.get_pet_config()

    last_fed_html = ""
    for pet in pet_config.keys():
        fed_time = last_fed.get(pet, "Never")
        last_fed_html += f"""
        <div class="last-fed-item">
            <div class="last-fed-name">{pet}</div>
            <div>Last fed: {fed_time}</div>
        </div>
        """

    body = f"""
    <div class="grid">
        <section class="panel">
            <h2>LIVE FEED</h2>
            <div class="feed-box">
                <img src="{url_for('video_feed')}" alt="Live feed">
            </div>
        </section>

        <section class="panel">
            <h2>MANUAL FEEDING</h2>
            <div class="manual-card">
                <div class="manual-subtitle">Quick manual dispense</div>
                <form method="post" action="{url_for('manual_feed')}">
                    <button class="action-btn" type="submit">FEED NOW</button>
                </form>

                <div class="manual-meta">
                    <div class="manual-meta-item">Manual portion: {core.manual_feed_seconds:.1f}s</div>
                    <div class="manual-meta-item">Last manual feed: {manual_last}</div>
                </div>
            </div>
        </section>

        <section class="panel">
            <h2>PET PROFILES PANEL</h2>
            <div class="last-fed-list">
                {last_fed_html}
            </div>
            <div class="pet-actions">
                <a href="{url_for('pets_page')}">EDIT FEEDING RULES</a>
            </div>
        </section>

        <section class="panel">
            <h2>RECENT FEEDING ACTIVITY</h2>
            {render_recent_activity(limit=5)}
        </section>
    </div>
    """
    return render_page(body)


@app.route("/live")
@login_required
def live_feed_page():
    body = f"""
    <section class="panel">
        <h2>LIVE FEED</h2>
        <div class="feed-box">
            <img src="{url_for('video_feed')}" alt="Live feed">
        </div>
    </section>
    """
    return render_page(body)


@app.route("/manual")
@login_required
def manual_page():
    manual_last = core.get_last_manual_feed_time()

    body = f"""
    <section class="panel">
        <h2>MANUAL FEEDING</h2>
        <div class="manual-card">
            <div class="manual-subtitle">Quick manual dispense</div>
            <form method="post" action="{url_for('manual_feed')}">
                <button class="action-btn" type="submit">FEED NOW</button>
            </form>

            <div class="manual-meta">
                <div class="manual-meta-item">Manual portion: {core.manual_feed_seconds:.1f}s</div>
                <div class="manual-meta-item">Last manual feed: {manual_last}</div>
            </div>
        </div>
    </section>
    """
    return render_page(body)


@app.route("/manual_feed", methods=["POST"])
@login_required
def manual_feed():
    threading.Thread(target=core.manual_feed, daemon=True).start()
    return redirect(request.referrer or url_for("index"))


@app.route("/pets", methods=["GET", "POST"])
@login_required
def pets_page():
    if request.method == "POST":
        config = core.get_pet_config()

        for pet in config:
            config[pet]["dispense_seconds"] = float(request.form[f"{pet}_dispense"])
            minutes = int(request.form[f"{pet}_cooldown"])
            config[pet]["cooldown_seconds"] = minutes * 60
            config[pet]["confidence_threshold"] = float(request.form[f"{pet}_confidence"])

        core.update_pet_config(config)
        return redirect(url_for("pets_page"))

    config = core.get_pet_config()
    last_fed = core.get_last_fed_map()
    cards = ""

    for pet, settings in config.items():
        current_minutes = settings["cooldown_seconds"] // 60
        fed_time = last_fed.get(pet, "Never")

        cards += f"""
        <div class="pet-card">
            <h2>{pet}</h2>
            <div class="pet-meta">Last fed: {fed_time}</div>

            <label>Dispense amount</label>
            <select name="{pet}_dispense">
        """
        for value, label in DISPENSE_OPTIONS:
            selected = "selected" if float(settings["dispense_seconds"]) == float(value) else ""
            cards += f'<option value="{value}" {selected}>{label}</option>'

        cards += f"""
            </select>

            <label style="margin-top:14px;">Cooldown interval</label>
            <select name="{pet}_cooldown">
        """
        for minutes, label in COOLDOWN_OPTIONS:
            selected = "selected" if current_minutes == minutes else ""
            cards += f'<option value="{minutes}" {selected}>{label}</option>'

        cards += f"""
            </select>

            <label style="margin-top:14px;">Confidence threshold</label>
            <select name="{pet}_confidence">
        """
        for value, label in CONFIDENCE_OPTIONS:
            selected = "selected" if float(settings["confidence_threshold"]) == float(value) else ""
            cards += f'<option value="{value}" {selected}>{label}</option>'

        cards += """
            </select>
        </div>
        """

    body = f"""
    <form method="post">
        <div class="form-grid">
            {cards}
        </div>
        <div class="save-row">
            <button class="action-btn" type="submit" style="font-size:20px; padding:12px 20px;">SAVE CHANGES</button>
        </div>
    </form>
    """
    return render_page(body)


@app.route("/history")
@login_required
def history_page():
    body = f"""
    <section class="panel">
        <h2>FEEDING HISTORY LOG</h2>
        {render_history_table(limit=50)}
    </section>
    """
    return render_page(body)


@app.route("/video_feed")
@login_required
def video_feed():
    def generate():
        import time
        last_frame = None

        while True:
            frame = core.get_latest_jpeg()

            if frame is None:
                time.sleep(0.02)
                continue

            annotated_frame = annotate_jpeg_frame(frame)

            if annotated_frame == last_frame:
                time.sleep(0.01)
                continue

            last_frame = annotated_frame

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-store, no-cache, must-revalidate, max-age=0\r\n"
                b"Pragma: no-cache\r\n\r\n" + annotated_frame + b"\r\n"
            )

    response = Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["X-Accel-Buffering"] = "no"

    return response


if __name__ == "__main__":
    try:
        core.start()
        app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
    finally:
        core.stop()
