import cv2
import mediapipe as mp
import json
import os
from datetime import datetime

# ══════════════════════════════════════════
# SHELFEYE - FIXED TOP/MIDDLE/BOTTOM DETECTION
# ══════════════════════════════════════════

ZONE_MAP = {
    "LEFT_TOP":      "Aisle 1 - Top Shelf",
    "LEFT_CENTER":   "Aisle 1 - Eye Level",
    "LEFT_BOTTOM":   "Aisle 1 - Bottom Shelf",
    "CENTER_TOP":    "Aisle 2 - Top Shelf",
    "CENTER_CENTER": "Aisle 2 - Eye Level",
    "CENTER_BOTTOM": "Aisle 2 - Bottom Shelf",
    "RIGHT_TOP":     "Aisle 3 - Top Shelf",
    "RIGHT_CENTER":  "Aisle 3 - Eye Level",
    "RIGHT_BOTTOM":  "Aisle 3 - Bottom Shelf",
}

AISLE_NAMES = ["Aisle 1", "Aisle 2", "Aisle 3"]

COLORS = {
    "LEFT":   (255, 100,   0),
    "CENTER": (  0, 200,   0),
    "RIGHT":  (  0, 100, 255),
}

VERT_COLORS = {
    "TOP":    (255, 50,  50),
    "CENTER": (  0, 200,  0),
    "BOTTOM": ( 50, 50, 255),
}

# ── THRESHOLDS ────────────────────────────
# These control sensitivity of detection
# Increase number = less sensitive (harder to trigger)
# Decrease number = more sensitive (easier to trigger)

YAW_THRESHOLD   = 0.04   # left/right sensitivity
PITCH_THRESHOLD = 0.04   # up/down sensitivity

# Visit thresholds
PASS_THRESHOLD   = 3
GLANCE_THRESHOLD = 8
BROWSE_THRESHOLD = 30
STABILITY_FRAMES = 20

# ── CALIBRATION ───────────────────────────
# This stores the neutral position of YOUR head
# System calibrates automatically in first 30 frames

calibration_frames  = []
calibrated          = False
neutral_yaw         = 0.0
neutral_pitch       = 0.0
CALIBRATION_NEEDED  = 30   # frames to calibrate

def calibrate(yaw, pitch):
    global calibrated, neutral_yaw, neutral_pitch
    calibration_frames.append((yaw, pitch))
    if len(calibration_frames) >= CALIBRATION_NEEDED:
        neutral_yaw   = sum(f[0] for f in calibration_frames) / len(calibration_frames)
        neutral_pitch = sum(f[1] for f in calibration_frames) / len(calibration_frames)
        calibrated    = True
        print(f"Calibrated! Neutral yaw: {neutral_yaw:.3f}  pitch: {neutral_pitch:.3f}")

def get_head_direction(face_landmarks):
    """
    Uses multiple landmarks spread across the face
    for much more accurate pitch detection
    """
    lm = face_landmarks.landmark

    # ── HORIZONTAL (YAW) ──────────────────
    # Use left ear vs right ear — widest horizontal points
    # Landmark 234 = left face edge
    # Landmark 454 = right face edge
    left_edge  = lm[234].x
    right_edge = lm[454].x
    face_width = right_edge - left_edge

    # Nose tip position relative to face center
    nose_x     = lm[4].x
    face_center_x = (left_edge + right_edge) / 2
    yaw_raw    = nose_x - face_center_x

    # Normalize by face width so distance doesnt matter
    yaw = yaw_raw / face_width if face_width > 0 else 0

    # ── VERTICAL (PITCH) ──────────────────
    # Use forehead (10) vs chin (152) — tallest vertical points
    # This gives a MUCH bigger range of motion to detect
    forehead_y = lm[10].y    # top of forehead
    chin_y     = lm[152].y   # bottom of chin
    face_height = chin_y - forehead_y

    # Nose position relative to face vertical center
    nose_y        = lm[4].y
    face_center_y = (forehead_y + chin_y) / 2
    pitch_raw     = nose_y - face_center_y

    # Normalize by face height
    pitch = pitch_raw / face_height if face_height > 0 else 0

    return yaw, pitch

# ── PERSON TRACKER ────────────────────────
class PersonTracker:
    def __init__(self):
        self.next_id     = 1
        self.active      = {}
        self.completed   = []
        self.MAX_LOST    = 45

    def update(self, detections):
        now         = datetime.now()
        matched_ids = set()

        for (cx, cy, zone) in detections:
            matched_id = self._match(cx, cy)

            if matched_id is not None:
                person = self.active[matched_id]
                person["cx"]          = cx
                person["cy"]          = cy
                person["last_seen"]   = now
                person["lost_frames"] = 0
                matched_ids.add(matched_id)

                if zone == person["candidate_zone"]:
                    person["stability"] += 1
                else:
                    person["candidate_zone"] = zone
                    person["stability"]       = 0

                if person["stability"] >= STABILITY_FRAMES:
                    if person["confirmed_zone"] != zone:
                        self._close_zone_visit(person, now)
                        person["confirmed_zone"] = zone
                        person["zone_start"]     = now
            else:
                new_id = self.next_id
                self.next_id += 1
                self.active[new_id] = {
                    "id":             new_id,
                    "cx": cx, "cy": cy,
                    "first_seen":     now,
                    "last_seen":      now,
                    "lost_frames":    0,
                    "candidate_zone": zone,
                    "confirmed_zone": None,
                    "zone_start":     now,
                    "stability":      0,
                    "visits":         [],
                }
                matched_ids.add(new_id)

        to_remove = []
        for pid in list(self.active.keys()):
            if pid not in matched_ids:
                self.active[pid]["lost_frames"] += 1
                if self.active[pid]["lost_frames"] > self.MAX_LOST:
                    person = self.active[pid]
                    self._close_zone_visit(person, now)
                    self.completed.append(person)
                    to_remove.append(pid)
        for pid in to_remove:
            del self.active[pid]

    def _match(self, cx, cy, threshold=120):
        best_id   = None
        best_dist = threshold
        for pid, person in self.active.items():
            dist = ((person["cx"]-cx)**2 + (person["cy"]-cy)**2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_id   = pid
        return best_id

    def _close_zone_visit(self, person, now):
        if not person["confirmed_zone"] or not person["zone_start"]:
            return
        duration = (now - person["zone_start"]).total_seconds()
        aisle    = self._get_aisle(person["confirmed_zone"])
        if duration < PASS_THRESHOLD or not aisle:
            return
        if duration < GLANCE_THRESHOLD:
            category = "glance"
        elif duration < BROWSE_THRESHOLD:
            category = "browse"
        else:
            category = "dwell"
        person["visits"].append({
            "aisle":    aisle,
            "zone":     person["confirmed_zone"],
            "duration": round(duration, 1),
            "category": category,
            "time":     now.strftime("%H:%M:%S")
        })

    def _get_aisle(self, zone_name):
        if not zone_name:
            return None
        for a in AISLE_NAMES:
            if a in zone_name:
                return a
        return None

    def get_aisle_stats(self):
        stats = {a: {"glance":0,"browse":0,"dwell":0,"unique_people":0}
                 for a in AISLE_NAMES}
        people_per_aisle = {a: set() for a in AISLE_NAMES}
        for person in list(self.active.values()) + self.completed:
            for visit in person["visits"]:
                aisle = visit["aisle"]
                stats[aisle][visit["category"]] += 1
                people_per_aisle[aisle].add(person["id"])
        for aisle in AISLE_NAMES:
            stats[aisle]["unique_people"] = len(people_per_aisle[aisle])
        return stats

    def get_active_count(self): return len(self.active)
    def get_total_seen(self):   return self.next_id - 1


# ── SETUP ─────────────────────────────────
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)
face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.4
)

tracker     = PersonTracker()
cap         = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frame_count = 0
log_file    = "attention_log.json"

print("ShelfEye running...")
print("CALIBRATING — look straight at camera for 3 seconds")
print("Press Q to quit | S to save | R to reset | C to recalibrate")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_count += 1
    h, w = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    det_results  = face_detection.process(rgb)
    mesh_results = face_mesh.process(rgb)

    detections = []

    if det_results.detections and mesh_results.multi_face_landmarks:
        for i, detection in enumerate(det_results.detections):
            bbox = detection.location_data.relative_bounding_box
            cx   = int((bbox.xmin + bbox.width/2)  * w)
            cy   = int((bbox.ymin + bbox.height/2) * h)

            zone = "Aisle 2 - Eye Level"

            if i < len(mesh_results.multi_face_landmarks):
                face = mesh_results.multi_face_landmarks[i]

                try:
                    yaw, pitch = get_head_direction(face)

                    # Calibrate first
                    if not calibrated:
                        calibrate(yaw, pitch)
                    else:
                        # Apply calibration offset
                        adj_yaw   = yaw   - neutral_yaw
                        adj_pitch = pitch - neutral_pitch

                        # Classify horizontal
                        if adj_yaw < -YAW_THRESHOLD:
                            h_zone = "LEFT"
                        elif adj_yaw > YAW_THRESHOLD:
                            h_zone = "RIGHT"
                        else:
                            h_zone = "CENTER"

                        # Classify vertical
                        if adj_pitch < -PITCH_THRESHOLD:
                            v_zone = "TOP"
                        elif adj_pitch > PITCH_THRESHOLD:
                            v_zone = "BOTTOM"
                        else:
                            v_zone = "CENTER"

                        zone = ZONE_MAP.get(
                            f"{h_zone}_{v_zone}",
                            "Aisle 2 - Eye Level"
                        )

                        # Draw on frame
                        color   = COLORS[h_zone]
                        v_color = VERT_COLORS[v_zone]

                        # Face box
                        x1 = int(bbox.xmin * w)
                        y1 = int(bbox.ymin * h)
                        bw = int(bbox.width * w)
                        bh = int(bbox.height * h)
                        cv2.rectangle(frame, (x1,y1), (x1+bw,y1+bh), color, 2)

                        # Zone label
                        cv2.putText(
                            frame, zone,
                            (cx - 80, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, color, 2
                        )

                        # Show raw angles for debugging
                        cv2.putText(
                            frame,
                            f"yaw:{adj_yaw:.3f} pitch:{adj_pitch:.3f}",
                            (cx - 80, cy + bh//2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (200,200,200), 1
                        )

                        # Vertical indicator bar on left side
                        bar_x  = 20
                        bar_y1 = 100
                        bar_y2 = 380
                        bar_h  = bar_y2 - bar_y1
                        cv2.rectangle(frame, (bar_x, bar_y1),
                                      (bar_x+20, bar_y2), (50,50,50), -1)

                        # Indicator position
                        norm_pitch  = max(-1, min(1, adj_pitch / 0.15))
                        indicator_y = int(bar_y1 + (norm_pitch+1)/2 * bar_h)
                        cv2.circle(frame, (bar_x+10, indicator_y),
                                   10, v_color, -1)

                        cv2.putText(frame, "TOP",
                                    (bar_x-5, bar_y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4, (255,100,100), 1)
                        cv2.putText(frame, "MID",
                                    (bar_x-5, (bar_y1+bar_y2)//2),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4, (100,255,100), 1)
                        cv2.putText(frame, "BOT",
                                    (bar_x-5, bar_y2+15),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4, (100,100,255), 1)

                except Exception as e:
                    pass

            detections.append((cx, cy, zone))

    tracker.update(detections)

    # ── TOP BAR ───────────────────────────
    cv2.rectangle(frame, (0,0), (w,85), (20,20,20), -1)

    if not calibrated:
        remaining = CALIBRATION_NEEDED - len(calibration_frames)
        cv2.putText(
            frame,
            f"CALIBRATING — look straight ahead ({remaining} frames left)",
            (15, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 220, 255), 2
        )
    else:
        cv2.putText(
            frame,
            f"People: {tracker.get_active_count()}   "
            f"Total seen: {tracker.get_total_seen()}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255,255,255), 2
        )
        cv2.putText(
            frame,
            "Look UP for top shelf | CENTER for eye level | DOWN for bottom",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (150,150,150), 1
        )

    # ── STATS PANEL ───────────────────────
    stats   = tracker.get_aisle_stats()
    panel_x = w - 230
    panel_y = 95
    cv2.rectangle(frame, (panel_x-10,panel_y),
                  (w-5, panel_y+200), (20,20,20), -1)
    cv2.putText(frame, "AISLE STATS",
                (panel_x, panel_y+22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255,255,0), 2)

    for i, aisle in enumerate(AISLE_NAMES):
        y  = panel_y + 48 + i*52
        st = stats[aisle]
        cv2.putText(frame, f"{aisle}:",
                    (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,200,255), 2)
        cv2.putText(
            frame,
            f"People:{st['unique_people']}  "
            f"G:{st['glance']} B:{st['browse']} D:{st['dwell']}",
            (panel_x, y+22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (180,180,180), 1
        )

    cv2.imshow("ShelfEye", frame)

    # Auto save every 5 seconds
    if frame_count % 150 == 0 and tracker.get_total_seen() > 0:
        with open(log_file, "w") as f:
            json.dump({
                "last_saved":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_people_seen": tracker.get_total_seen(),
                "active_now":        tracker.get_active_count(),
                "aisle_stats":       tracker.get_aisle_stats(),
            }, f, indent=2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        with open(log_file, "w") as f:
            json.dump({
                "last_saved":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_people_seen": tracker.get_total_seen(),
                "active_now":        tracker.get_active_count(),
                "aisle_stats":       tracker.get_aisle_stats(),
            }, f, indent=2)
        print("Saved!")
    elif key == ord('r'):
        tracker = PersonTracker()
        print("Reset!")
    elif key == ord('c'):
        calibration_frames.clear()
        calibrated = False
        print("Recalibrating...")

cap.release()
cv2.destroyAllWindows()


# ## How To Test It
# ```
# 1. Run the code
# 2. Wait for "Calibrated!" message in terminal
# 3. Look straight → should show CENTER CENTER
# 4. Look UP slowly → dot on bar moves up → shows TOP
# 5. Look DOWN slowly → dot moves down → shows BOTTOM
# 6. Turn head LEFT → shows LEFT
# 7. Turn head RIGHT → shows RIGHT
##
