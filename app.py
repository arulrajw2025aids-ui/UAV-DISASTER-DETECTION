import cv2
import os
import time
import random
import threading
import requests
import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from flask import Flask, Response, jsonify, render_template, request, redirect, url_for, session
from functools import wraps
from ultralytics import YOLO
from twilio.rest import Client as TwilioClient
from twilio.base.exceptions import TwilioRestException
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
CAMERA_SOURCE    = 0
IMAGE_FOLDER     = "phone_captured_images"
ALERT_SERVER_URL = "http://localhost:5000/alert"
CAPTURE_INTERVAL = 10
YOLO_EVERY       = 3
DB_PATH          = "uav_contacts.db"

# --- SECRETS (loaded from .env) ---
TWILIO_SID        = os.getenv("TWILIO_SID")
TWILIO_TOKEN      = os.getenv("TWILIO_TOKEN")
TWILIO_FROM       = os.getenv("TWILIO_FROM")
GOOGLE_CLIENT_ID  = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:5000/google_callback")

os.makedirs(IMAGE_FOLDER, exist_ok=True)

app            = Flask(__name__)
app.secret_key = os.getenv("APP_SECRET_KEY", "fallback_secret_key")
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
model          = YOLO("yolov8n.pt")

# -------------------------------------------------------
# DATABASE SETUP
# -------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT NOT NULL UNIQUE,
                password   TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS contacts (
                id         TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                phone      TEXT NOT NULL,
                role       TEXT NOT NULL,
                address    TEXT DEFAULT '',
                notes      TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS message_log (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                sent_to   TEXT NOT NULL,
                phone     TEXT NOT NULL,
                role      TEXT NOT NULL,
                message   TEXT NOT NULL,
                status    TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        conn.commit()
    print("Database initialized: {}".format(DB_PATH))

init_db()


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in") or not session.get("username"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# --- SHARED STATE ---
state = {
    "human_count":      0,
    "severity":         "NONE",
    "gps":              {"latitude": 0, "longitude": 0, "altitude": 0, "weather": {"temp": "--", "wind": "--", "code": "--"}},
    "last_alert_time":  "-",
    "alerts":           [],
    "total_detections": 0,
    "messages":         [],
}
lock          = threading.Lock()
raw_frame     = None
output_frame  = None
camera_active = False

# -------------------------------------------------------
# SMS SENDER via Twilio
# -------------------------------------------------------
twilio_client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)

def send_sms(phone, message):
    number = phone.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    e164 = None
    if number.startswith("+"):
        e164 = number
    elif number.startswith("91") and len(number) == 12 and number.isdigit():
        e164 = "+" + number
    elif number.startswith("0") and len(number) == 11:
        e164 = "+91" + number[1:]
    elif len(number) == 10 and number.isdigit():
        e164 = "+91" + number
    else:
        return False, "Invalid phone number: " + phone

    print("Sending SMS to: {}".format(e164))
    try:
        msg = twilio_client.messages.create(
            body=message,
            from_=TWILIO_FROM,
            to=e164
        )
        print("SMS sent to {} SID: {}".format(e164, msg.sid))
        return True, "Sent"
    except TwilioRestException as e:
        print("Twilio error for {}: {}".format(e164, e.msg))
        return False, e.msg
    except Exception as e:
        print("SMS error for {}: {}".format(e164, e))
        return False, str(e)

# -------------------------------------------------------
# GPS SIMULATION
# -------------------------------------------------------
def get_gps():
    lat = round(random.uniform(12.95, 13.05), 6)
    lon = round(random.uniform(77.55, 77.65), 6)
    alt = random.randint(30, 80)
    weather = {"temp": "--", "wind": "--", "code": "--"}
    try:
        w_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        w_data = requests.get(w_url, timeout=2).json()
        if "current_weather" in w_data:
            weather = {
                "temp": w_data["current_weather"].get("temperature", "--"),
                "wind": w_data["current_weather"].get("windspeed", "--"),
                "code": w_data["current_weather"].get("weathercode", "--")
            }
    except Exception:
        pass
    return {
        "latitude":  lat,
        "longitude": lon,
        "altitude":  alt,
        "weather":   weather
    }

def severity_level(humans):
    if humans == 0: return "NONE"
    if humans == 1: return "MEDIUM"
    return "HIGH"

def send_alert(humans, gps):
    sev = severity_level(humans)
    alert_data = {
        "humans_detected": humans,
        "severity":        sev,
        "gps_location":    gps,
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("ALERT:", alert_data)
    try:
        requests.post(ALERT_SERVER_URL, json=alert_data, timeout=1)
    except Exception:
        print("Alert server not reachable")

# -------------------------------------------------------
# CAPTURE THREAD
# -------------------------------------------------------
def capture_loop():
    global raw_frame, camera_active
    print("UAV Disaster Monitoring System Started")
    cap = None
    while True:
        if not camera_active:
            if cap is not None:
                cap.release()
                cap = None
                with lock:
                    raw_frame = None
            time.sleep(0.5)
            continue
        if cap is None:
            cap = cv2.VideoCapture(CAMERA_SOURCE)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                print("Unable to open laptop camera - retrying...")
                cap = None
                time.sleep(1)
                continue
            print("Laptop camera connected")
        ret, frame = cap.read()
        if not ret:
            print("Frame not received - reconnecting...")
            cap.release()
            cap = None
            time.sleep(0.5)
            continue
        frame = cv2.resize(frame, (640, 480))
        with lock:
            raw_frame = frame

# -------------------------------------------------------
# DETECTION THREAD
# -------------------------------------------------------
def detection_loop():
    global output_frame
    last_capture     = time.time()
    frame_idx        = 0
    last_boxes       = []
    last_human_count = 0

    while True:
        with lock:
            frame = raw_frame.copy() if raw_frame is not None else None
        if frame is None:
            time.sleep(0.01)
            continue

        frame_idx += 1

        if frame_idx % YOLO_EVERY == 0:
            results = model(frame, verbose=False, imgsz=320)
            last_boxes       = []
            last_human_count = 0
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:
                        last_human_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        last_boxes.append((x1, y1, x2, y2, conf))

        for (x1, y1, x2, y2, conf) in last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 80), 2)
            cv2.putText(frame, "Human {:.0%}".format(conf), (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2)

        sev   = severity_level(last_human_count)
        color = (0, 255, 80) if sev == "NONE" else (0, 165, 255) if sev == "MEDIUM" else (0, 0, 255)
        cv2.putText(frame, "Humans: {}  [{}]".format(last_human_count, sev), (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (15, 465),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        now = time.time()
        if last_human_count > 0 and (now - last_capture >= CAPTURE_INTERVAL):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = "{}/detection_{}.jpg".format(IMAGE_FOLDER, ts)
            cv2.imwrite(path, frame)
            print("Image saved: {}".format(path))
            gps = get_gps()
            send_alert(last_human_count, gps)
            alert = {
                "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "disaster_type": "Human Presence Detected - Natural Disaster Zone",
                "humans":       last_human_count,
                "severity":     sev,
                "gps":          gps,
                "image":        path,
            }
            with lock:
                state["human_count"]      = last_human_count
                state["severity"]         = sev
                state["disaster_type"]    = alert["disaster_type"]
                state["gps"]              = gps
                state["last_alert_time"]  = alert["timestamp"]
                state["alerts"].insert(0, alert)
                state["alerts"]           = state["alerts"][:50]
                state["total_detections"] += last_human_count
            last_capture = now

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with lock:
            output_frame = buf.tobytes()

# -------------------------------------------------------
# FRAME GENERATOR
# -------------------------------------------------------
def gen_frames():
    global output_frame
    while True:
        with lock:
            frame = output_frame
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.02)


# -------------------------------------------------------
# FLASK ROUTES
# -------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        with get_db() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE username=? AND password=?",
                (username, hash_password(password))
            ).fetchone()
        if user:
            global camera_active
            camera_active       = True
            session.permanent   = True
            session["logged_in"] = True
            session["username"]  = username
            return redirect(url_for("home"))
        error = "Invalid username or password."
    return render_template("login.html", error=error, google_client_id=GOOGLE_CLIENT_ID)


@app.route("/signup", methods=["POST"])
def signup():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    confirm  = request.form.get("confirm",  "").strip()
    if not username or not password:
        return render_template("login.html", signup_error="Username and password are required.", show_signup=True, google_client_id=GOOGLE_CLIENT_ID)
    if password != confirm:
        return render_template("login.html", signup_error="Passwords do not match.", show_signup=True, google_client_id=GOOGLE_CLIENT_ID)
    if len(password) < 6:
        return render_template("login.html", signup_error="Password must be at least 6 characters.", show_signup=True, google_client_id=GOOGLE_CLIENT_ID)
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO users (username,password,created_at) VALUES (?,?,?)",
                (username, hash_password(password), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
        global camera_active
        camera_active        = True
        session.clear()
        session["logged_in"] = True
        session["username"]  = username
        session.modified     = True
        return redirect(url_for("home"))
    except sqlite3.IntegrityError:
        return render_template("login.html", signup_error="Username already exists.", show_signup=True, google_client_id=GOOGLE_CLIENT_ID)


@app.route("/google_callback")
def google_callback():
    code = request.args.get("code")
    if not code:
        return redirect(url_for("login"))
    try:
        import urllib.request, urllib.parse, json as _json
        GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
        redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:5000/google_callback")
        token_data = urllib.parse.urlencode({
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code"
        }).encode()
        token_req = urllib.request.Request("https://oauth2.googleapis.com/token", data=token_data, method="POST")
        token_req.add_header("Content-Type", "application/x-www-form-urlencoded")
        with urllib.request.urlopen(token_req) as resp:
            token_json = _json.loads(resp.read())
        access_token = token_json.get("access_token", "")
        user_req = urllib.request.Request(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": "Bearer " + access_token}
        )
        with urllib.request.urlopen(user_req) as resp:
            user_info = _json.loads(resp.read())
        email    = user_info.get("email", "")
        name     = user_info.get("name", email.split("@")[0])
        username = name.replace(" ", "_").lower()
        with get_db() as conn:
            user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
            if not user:
                conn.execute(
                    "INSERT INTO users (username,password,created_at) VALUES (?,?,?)",
                    (username, hash_password(email + "_google"), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                )
                conn.commit()
        global camera_active
        camera_active        = True
        session["logged_in"] = True
        session["username"]  = username
        return redirect(url_for("home"))
    except Exception as e:
        return redirect(url_for("login"))

@app.route("/logout")
def logout():
    camera_active = False
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def home():
    return render_template("index.html", username=session.get("username"))


@app.route("/dashboard")
@login_required
def dashboard():
    global camera_active
    camera_active = True
    return render_template("dashboard.html", username=session.get("username"))


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
@login_required
def status():
    with lock:
        return jsonify({
            "human_count":      state["human_count"],
            "severity":         state["severity"],
            "gps":              state["gps"],
            "last_alert_time":  state["last_alert_time"],
            "total_detections": state["total_detections"],
            "alert_count":      len(state["alerts"]),
        })


@app.route("/alerts")
@login_required
def alerts():
    with lock:
        return jsonify(state["alerts"])


@app.route("/alert", methods=["POST"])
def receive_alert():
    data = request.json
    if data:
        data.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        with lock:
            state["alerts"].insert(0, data)
            state["alerts"] = state["alerts"][:50]
    return jsonify({"status": "received"})


@app.route("/contacts", methods=["GET"])
@login_required
def get_contacts():
    role = request.args.get("role", "").strip()
    with get_db() as conn:
        if role:
            rows = conn.execute(
                "SELECT * FROM contacts WHERE LOWER(role)=? ORDER BY role, name",
                (role.lower(),)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM contacts ORDER BY role, name"
            ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/contacts", methods=["POST"])
@login_required
def add_contact():
    data = request.json
    if not data or not data.get("name") or not data.get("phone"):
        return jsonify({"error": "name and phone required"}), 400
    contact = {
        "id":         datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "name":       data["name"].strip(),
        "phone":      data["phone"].strip(),
        "role":       data.get("role", "Rescue Team").strip(),
        "address":    data.get("address", "").strip(),
        "notes":      data.get("notes", "").strip(),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with get_db() as conn:
        conn.execute(
            "INSERT INTO contacts (id,name,phone,role,address,notes,created_at) VALUES (?,?,?,?,?,?,?)",
            (contact["id"], contact["name"], contact["phone"],
             contact["role"], contact["address"], contact["notes"], contact["created_at"])
        )
        conn.commit()
    return jsonify({"status": "added", "contact": contact})


@app.route("/contacts/<cid>", methods=["DELETE"])
@login_required
def delete_contact(cid):
    with get_db() as conn:
        conn.execute("DELETE FROM contacts WHERE id=?", (cid,))
        conn.commit()
    return jsonify({"status": "deleted"})


@app.route("/contacts/<cid>", methods=["PUT"])
@login_required
def update_contact(cid):
    data = request.json
    with get_db() as conn:
        conn.execute(
            "UPDATE contacts SET name=?, phone=?, role=?, address=?, notes=? WHERE id=?",
            (data.get("name"), data.get("phone"), data.get("role"),
             data.get("address", ""), data.get("notes", ""), cid)
        )
        conn.commit()
    return jsonify({"status": "updated"})


@app.route("/contacts/stats", methods=["GET"])
@login_required
def contact_stats():
    with get_db() as conn:
        total      = conn.execute("SELECT COUNT(*) FROM contacts").fetchone()[0]
        victims    = conn.execute("SELECT COUNT(*) FROM contacts WHERE LOWER(role)='victim'").fetchone()[0]
        volunteers = conn.execute("SELECT COUNT(*) FROM contacts WHERE LOWER(role)='volunteer'").fetchone()[0]
        rescue     = conn.execute("SELECT COUNT(*) FROM contacts WHERE LOWER(role)='rescue team'").fetchone()[0]
    return jsonify({"total": total, "victims": victims, "volunteers": volunteers, "rescue_team": rescue})


@app.route("/send_message", methods=["POST"])
@login_required
def send_message():
    data    = request.json
    message = data.get("message", "").strip()
    target  = data.get("target", "all")
    if not message:
        return jsonify({"error": "message required"}), 400

    with get_db() as conn:
        if target == "all":
            rows = conn.execute("SELECT * FROM contacts ORDER BY role, name").fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM contacts WHERE LOWER(role)=? ORDER BY name",
                (target.lower(),)
            ).fetchall()
        contacts = [dict(r) for r in rows]

    if not contacts:
        return jsonify({"status": "ok", "sent_to": [], "failed": [], "total": 0})

    sent_names  = []
    failed_names = []

    def send_one(c):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = "[UAV ALERT] Role:{} | {} | {}".format(c["role"], message, ts)
        ok, status_msg = send_sms(c["phone"], body)
        final_status = "Sent" if ok else "Failed: " + status_msg
        with get_db() as db:
            db.execute(
                "INSERT INTO message_log (sent_to,phone,role,message,status,timestamp) VALUES (?,?,?,?,?,?)",
                (c["name"], c["phone"], c["role"], message, final_status, ts)
            )
            db.commit()
        with lock:
            state["messages"].insert(0, {
                "timestamp": ts, "to": c["name"], "phone": c["phone"],
                "role": c["role"], "message": message, "status": final_status,
            })
        if ok:
            sent_names.append(c["name"])
        else:
            failed_names.append(c["name"])

    with ThreadPoolExecutor(max_workers=min(10, len(contacts))) as executor:
        executor.map(send_one, contacts)
    with lock:
        state["messages"] = state["messages"][:100]

    return jsonify({"status": "ok", "sent_to": sent_names, "failed": failed_names, "total": len(contacts)})


@app.route("/send_shortage_alert", methods=["POST"])
@login_required
def send_shortage_alert():
    data     = request.json
    res_name = data.get("name", "Unknown")
    res_cat  = data.get("cat", "")
    qty      = data.get("qty", 0)
    min_qty  = data.get("min", 0)
    loc      = data.get("loc", "")

    message_en = (
        "[SHORTAGE ALERT] {} {} is critically low! "
        "Available: {} | Minimum Required: {} | Location: {}. "
        "Immediate resupply needed."
    ).format(res_cat, res_name, qty, min_qty, loc or "N/A")

    with get_db() as conn:
        contacts = [dict(r) for r in conn.execute("SELECT * FROM contacts").fetchall()]

    if not contacts:
        return jsonify({"status": "no_contacts", "sent": 0})

    sent, failed = [], []

    def send_one(c):
        ok, _ = send_sms(c["phone"], message_en)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with get_db() as db:
            db.execute(
                "INSERT INTO message_log (sent_to,phone,role,message,status,timestamp) VALUES (?,?,?,?,?,?)",
                (c["name"], c["phone"], c["role"], message_en, "Sent" if ok else "Failed", ts)
            )
            db.commit()
        (sent if ok else failed).append(c["name"])

    with ThreadPoolExecutor(max_workers=min(10, len(contacts))) as executor:
        executor.map(send_one, contacts)

    return jsonify({"status": "ok", "sent": len(sent), "failed": len(failed)})



@app.route("/messages/clear", methods=["DELETE"])
@login_required
def clear_messages():
    with get_db() as conn:
        conn.execute("DELETE FROM message_log")
        conn.commit()
    with lock:
        state["messages"] = []
    return jsonify({"status": "cleared"})


@app.route("/messages", methods=["GET"])
@login_required
def get_messages():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM message_log ORDER BY id DESC LIMIT 100"
        ).fetchall()
    return jsonify([dict(r) for r in rows])


# -------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------
if __name__ == "__main__":
    threading.Thread(target=capture_loop,   daemon=True).start()
    threading.Thread(target=detection_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
