# -*- coding: utf-8 -*-
"""
===========================================================================aa
 DDD — Driver Drowsiness Detection
 Détection de somnolence en temps réel via webcam
===========================================================================
 Modèle   : DDD_model_augmented.h5 / .keras (EfficientNetB0, 4 classes)
 Classes  : 0=Closed  1=Open  2=no_yawn  3=yawn

 Fonctionnalités :
   - Détection yeux fermés + bâillement en temps réel
   - Score de fatigue progressif avec jauge visuelle
   - Alarme visuelle (cadre rouge clignotant) + sonore (bip)
   - Conseils de sécurité adaptés au niveau de fatigue
   - Clustering d'alarmes → enregistrement heures à risque
   - Affichage "Allez vous reposer !" aux heures de risque

 Commandes :
   Q   — Quitter
===========================================================================
"""

import cv2
import numpy as np
from datetime import datetime
import time
import json
import os
import threading

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, 'output_plots')
_KERAS_PATH = os.path.join(OUTPUT_DIR, 'DDD_model_augmented.keras')
_H5_PATH    = os.path.join(OUTPUT_DIR, 'DDD_model_augmented.h5')

# Choisir le modèle : .keras prioritaire si valide (>100 Ko), sinon .h5
MODEL_PATH = (_KERAS_PATH
              if os.path.exists(_KERAS_PATH) and os.path.getsize(_KERAS_PATH) > 100_000
              else _H5_PATH)

# Classes (ordre alphabétique Keras)
CLASS_NAMES = ['Closed', 'Open', 'no_yawn', 'yawn']
CLOSED      = 0
OPEN        = 1
NO_YAWN     = 2
YAWN        = 3

# Paramètres détection
IMG_SIZE           = 100
ALARM_THRESHOLD    = 12      # score de fatigue déclenchant l'alarme
CLOSED_FRAMES_MIN  = 3       # frames consécutives yeux fermés avant +1
YAWN_FRAMES_MIN    = 3       # frames consécutives bâillement avant +2

# Clustering d'alarmes
CLUSTER_WIN_MIN = 1.5        # écart min entre 2 alarmes (s)
CLUSTER_WIN_MAX = 5.0        # écart max entre 2 alarmes (s)

# Heures de repos
REST_FILE        = os.path.join(BASE_DIR, 'rest_times.json')
REST_TOLERANCE   = 5         # tolérance ±5 min autour de l'heure enregistrée
REST_DISPLAY_SEC = 10        # durée d'affichage du message repos (s)
REST_MAX_SHOWS   = 2         # nombre max d'affichages par heure de risque

# Alarme sonore
SOUND_ENABLED = True
BEEP_FREQ     = 2500
BEEP_DURATION = 300
BEEP_COOLDOWN = 3.0          # secondes entre 2 bips

# ═══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES RESSOURCES
# ═══════════════════════════════════════════════════════════════════════════════

WEIGHTS_PATH = os.path.join(OUTPUT_DIR, 'DDD_weights.h5')

print("[INFO] Chargement du modele...")
# Priorité 1 : modèle complet .keras (architecture + poids)
if os.path.exists(_KERAS_PATH) and os.path.getsize(_KERAS_PATH) > 100_000:
    model = load_model(_KERAS_PATH)
# Priorité 2 : modèle complet .h5
elif os.path.exists(_H5_PATH) and os.path.getsize(_H5_PATH) > 100_000:
    model = load_model(_H5_PATH)
# Priorité 3 : poids seuls → reconstruction avec Functional API (BatchNorm)
elif os.path.exists(WEIGHTS_PATH) and os.path.getsize(WEIGHTS_PATH) > 100_000:
    from tensorflow.keras import Model, Input
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
    from tensorflow.keras.applications import EfficientNetB0
    inputs     = Input(shape=(100, 100, 3))
    base_model = EfficientNetB0(weights='imagenet', include_top=False,
                                input_shape=(100, 100, 3))
    base_model.trainable = False
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(4, activation='softmax')(x)
    model   = Model(inputs, outputs)
    # Load weights manually — the h5 weight_names ordering is jumbled
    # (a TF 2.11 known issue), so we match by name instead of relying
    # on Keras's positional loader.
    import h5py
    with h5py.File(WEIGHTS_PATH, 'r') as f:
        wdict = {}
        def _collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                wdict[name] = np.array(obj[()])
        f.visititems(_collect)
    # Map h5 paths → model weight names
    model_weight_map = {w.name: w for w in model.weights}
    for h5_key, arr in wdict.items():
        parts = h5_key.split('/')
        if parts[0] == 'efficientnetb0':
            model_name = '/'.join(parts[1:])
        else:
            model_name = parts[0] + '/' + parts[-1]
        if model_name in model_weight_map:
            model_weight_map[model_name].assign(arr)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
else:
    print(f"[ERROR] Modele introuvable : {WEIGHTS_PATH}")
    print(f"        Lancez d'abord : python train_ddd_model.py")
    exit(1)
print(f"[INFO] Modele charge — entree : {model.input_shape}")

# Cascades Haar
face_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades +
                                      'haarcascade_frontalface_default.xml')
leye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades +
                                      'haarcascade_lefteye_2splits.xml')
reye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades +
                                      'haarcascade_righteye_2splits.xml')


# ═══════════════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

def beep():
    if SOUND_ENABLED:
        try:
            import winsound
            winsound.Beep(BEEP_FREQ, BEEP_DURATION)
        except Exception:
            pass


def load_rest_times():
    if os.path.exists(REST_FILE):
        with open(REST_FILE, 'r') as f:
            return json.load(f)
    return []


def save_rest_time(hhmm: str):
    times = load_rest_times()
    if hhmm not in times:
        times.append(hhmm)
        with open(REST_FILE, 'w') as f:
            json.dump(times, f)
        print(f"[INFO] Heure de risque enregistrée : {hhmm}")


def is_rest_time_now():
    now   = datetime.now()
    times = load_rest_times()
    for hhmm in times:
        h, m    = map(int, hhmm.split(':'))
        risk_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if abs((now - risk_dt).total_seconds()) <= REST_TOLERANCE * 60:
            return True
    return False


# ─── Prédiction ────────────────────────────────────────────────────────────────

def preprocess_roi(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi = roi.astype('float32')
    roi = preprocess_input(roi)
    return roi.reshape((1, IMG_SIZE, IMG_SIZE, 3))


def predict_class(roi_bgr):
    roi = preprocess_roi(roi_bgr)
    if roi is None:
        return -1
    return int(np.argmax(model.predict(roi, verbose=0), axis=-1)[0])


# ─── Détection bouche ─────────────────────────────────────────────────────────

def detect_yawn_roi(frame, faces):
    """
    Utilise le visage détecté pour la prédiction yawn/no_yawn.
    Le modèle est entraîné sur des images de visage (pas des crops bouche),
    donc on lui envoie la moitié inférieure du visage (zone bouche+menton).
    Retourne (roi_bgr, (x,y,w,h), source) ou (None, None, None)
    """
    for (fx, fy, fw, fh) in faces:
        # Moitié inférieure du visage (40-100% hauteur) → correspond aux données d'entraînement
        y1 = fy + int(fh * 0.40)
        y2 = fy + fh
        x1 = fx + int(fw * 0.05)
        x2 = fx + fw - int(fw * 0.05)
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            return roi, (x1, y1, x2 - x1, y2 - y1), "face"
    return None, None, None


# ─── Conseils de sécurité ─────────────────────────────────────────────────────

ADVICE_LIST = [
    "Faites une pause de 15 min toutes les 2h de conduite.",
    "Buvez du café ou du thé — la caféine aide temporairement.",
    "Si possible, changez de conducteur.",
    "Aérez l'habitacle, baissez la vitre.",
    "Étirez-vous et marchez quelques minutes.",
    "Évitez de conduire entre 2h et 6h du matin.",
    "Dormez au moins 7h avant de prendre la route.",
]

LEVEL_LABELS = [
    (0,  "ATTENTIF",           (0, 220, 0)),
    (5,  "FATIGUE LEGERE",     (0, 255, 255)),
    (8,  "FATIGUE MODEREE",    (0, 165, 255)),
    (12, "SOMNOLENCE !",       (0, 0, 255)),
]


def get_status(score):
    for threshold, label, color in reversed(LEVEL_LABELS):
        if score >= threshold:
            return label, color
    return LEVEL_LABELS[0][1], LEVEL_LABELS[0][2]


def get_advice(score):
    if score < 5:
        return None
    idx = min((score - 5) // 3, len(ADVICE_LIST) - 1)
    return ADVICE_LIST[idx]


# ─── HUD ───────────────────────────────────────────────────────────────────────

def draw_bar(frame, value, max_val, x, y, label, color):
    bw, bh = 150, 14
    fill = int(min(value / max(max_val, 1), 1.0) * bw)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (40, 40, 40), -1)
    cv2.rectangle(frame, (x, y), (x + fill, y + bh), color, -1)
    cv2.putText(frame, f"{label}: {value}", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1)


def centered_text(frame, text, y_frac, scale, color, thickness=2):
    h, w  = frame.shape[:2]
    sz, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(frame, text, ((w - sz[0]) // 2, int(h * y_frac)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_hud(frame, fatigue_score, eye_label, eye_color, yawn_label,
             yawn_color, mouth_src, alarm_triggered, alarm_threshold, now_dt):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    # Barre sombre en bas
    cv2.rectangle(frame, (0, h - 115), (w, h), (20, 20, 20), -1)

    # Statut fatigue
    status_label, status_color = get_status(fatigue_score)
    src_tag = f"[{mouth_src}]" if mouth_src else "[?]"

    cv2.putText(frame, f"Eyes  : {eye_label}",
                (10, h - 92), font, 0.85, eye_color, 1)
    cv2.putText(frame, f"Mouth : {yawn_label}  {src_tag}",
                (10, h - 70), font, 0.85, yawn_color, 1)
    cv2.putText(frame, f"Statut: {status_label}",
                (10, h - 48), font, 0.80, status_color, 1)
    cv2.putText(frame, f"Score : {fatigue_score}  [Alarme > {alarm_threshold}]",
                (10, h - 26), font, 0.70, (220, 220, 220), 1)

    # Conseil
    advice = get_advice(fatigue_score)
    if advice:
        cv2.putText(frame, f"Conseil: {advice}",
                    (10, h - 5), font, 0.50, (0, 255, 255), 1)

    # Jauge
    bar_color = (0, 0, 255) if alarm_triggered else (0, 200, 255)
    draw_bar(frame, fatigue_score, alarm_threshold + 10,
             w - 185, h - 105, "Fatigue", bar_color)

    # Horloge
    cv2.putText(frame, now_dt.strftime("%H:%M:%S"),
                (w - 110, 25), font, 0.9, (200, 200, 200), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# BOUCLE PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════════

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Impossible d'ouvrir la caméra.")
    exit(1)

# États
fatigue_score      = 0
closed_frame_count = 0
yawn_frame_count   = 0
thicc              = 2
alarm_timestamps   = []
in_alarm           = False
cluster_recorded   = False
rest_show_count    = 0
rest_show_start    = None
last_rest_key      = None
last_beep_time     = 0

print("[INFO] Detection lancee — appuyez sur 'q' pour quitter.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w   = frame.shape[:2]
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    now_ts = time.time()
    now_dt = datetime.now()

    # ── Détections Haar ────────────────────────────────────────────────────────
    faces      = face_cascade.detectMultiScale(gray, 1.1, 5)
    right_eyes = reye_cascade.detectMultiScale(gray)
    left_eyes  = leye_cascade.detectMultiScale(gray)

    rpred      = OPEN
    lpred      = OPEN
    ypred      = NO_YAWN
    r_detected = False
    l_detected = False

    # Œil droit
    for (x, y, bw, bh) in right_eyes:
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 200, 0), 1)
        p = predict_class(frame[y:y + bh, x:x + bw])
        if p != -1:
            rpred, r_detected = p, True
        break

    # Œil gauche
    for (x, y, bw, bh) in left_eyes:
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 200, 255), 1)
        p = predict_class(frame[y:y + bh, x:x + bw])
        if p != -1:
            lpred, l_detected = p, True
        break

    # Bâillement — utiliser la zone visage (pas un crop bouche)
    yawn_roi, yawn_rect, mouth_src = detect_yawn_roi(frame, faces)
    mouth_src = mouth_src or ""
    if yawn_roi is not None:
        mx, my, mw, mh = yawn_rect
        cv2.rectangle(frame, (mx, my), (mx + mw, my + mh),
                       (200, 0, 255), 2)
        p = predict_class(yawn_roi)
        if p != -1 and p in (YAWN, NO_YAWN):
            ypred = p

    # Visage
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 1)
        break

    # ══════════════════════════════════════════════════════════════════════════
    # LOGIQUE DE SCORE
    # ══════════════════════════════════════════════════════════════════════════
    both_closed = ((rpred == CLOSED or not r_detected) and
                   (lpred == CLOSED or not l_detected) and
                   (r_detected or l_detected))
    either_open = not both_closed
    is_yawning  = (ypred == YAWN)

    # Décroissance si état normal
    if either_open and not is_yawning:
        fatigue_score = max(0, fatigue_score - 1)

    # Yeux fermés : compter frames consécutives
    if both_closed:
        closed_frame_count += 1
        if closed_frame_count >= CLOSED_FRAMES_MIN:
            fatigue_score += 1
    else:
        closed_frame_count = 0

    # Bâillement : compter frames consécutives
    if is_yawning:
        yawn_frame_count += 1
        if yawn_frame_count >= YAWN_FRAMES_MIN:
            fatigue_score += 2
    else:
        yawn_frame_count = 0

    # ══════════════════════════════════════════════════════════════════════════
    # ALARME
    # ══════════════════════════════════════════════════════════════════════════
    alarm_triggered = fatigue_score > ALARM_THRESHOLD

    if alarm_triggered and not in_alarm:
        in_alarm = True
        alarm_timestamps.append(now_ts)
        print(f"[ALARM] {now_dt.strftime('%H:%M:%S')} — score={fatigue_score}")

        # Bip sonore (max 1 toutes les BEEP_COOLDOWN secondes)
        if now_ts - last_beep_time > BEEP_COOLDOWN:
            threading.Thread(target=beep, daemon=True).start()
            last_beep_time = now_ts

    if not alarm_triggered:
        in_alarm = False

    # ── Clustering d'alarmes ───────────────────────────────────────────────────
    if alarm_triggered and len(alarm_timestamps) >= 2 and not cluster_recorded:
        gap = alarm_timestamps[-1] - alarm_timestamps[-2]
        if CLUSTER_WIN_MIN <= gap <= CLUSTER_WIN_MAX:
            hhmm = now_dt.strftime("%H:%M")
            save_rest_time(hhmm)
            cluster_recorded = True
            print(f"[CLUSTER] Ecart {gap:.2f}s → heure : {hhmm}")
        elif gap > CLUSTER_WIN_MAX:
            alarm_timestamps = [alarm_timestamps[-1]]

    if len(alarm_timestamps) > 2:
        alarm_timestamps = [alarm_timestamps[-1]]

    alarm_timestamps = [t for t in alarm_timestamps if now_ts - t <= CLUSTER_WIN_MAX]

    if not alarm_triggered and len(alarm_timestamps) == 0:
        cluster_recorded = False

    # ══════════════════════════════════════════════════════════════════════════
    # AFFICHAGE
    # ══════════════════════════════════════════════════════════════════════════
    eye_label  = "Open" if either_open else "Closed"
    yawn_label = "yawn" if is_yawning else "no_yawn"
    eye_color  = (0, 255, 120) if either_open else (0, 0, 255)
    yawn_color = (0, 100, 255) if is_yawning else (0, 255, 120)

    # HUD (barre info + jauge + horloge)
    draw_hud(frame, fatigue_score, eye_label, eye_color, yawn_label,
             yawn_color, mouth_src, alarm_triggered, ALARM_THRESHOLD, now_dt)

    # ── Message "Allez vous reposer" aux heures de risque ──────────────────────
    if is_rest_time_now():
        current_key = now_dt.strftime("%H:%M")

        if last_rest_key != current_key:
            last_rest_key   = current_key
            rest_show_count = 0
            rest_show_start = None

        if rest_show_count < REST_MAX_SHOWS and rest_show_start is None:
            rest_show_start  = now_ts
            rest_show_count += 1
            print(f"[REST] Affichage {rest_show_count}/{REST_MAX_SHOWS}")

        if rest_show_start is not None:
            elapsed = now_ts - rest_show_start
            if elapsed <= REST_DISPLAY_SEC:
                remaining = int(REST_DISPLAY_SEC - elapsed) + 1
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h // 2 - 80), (w, h // 2 + 80),
                              (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
                centered_text(frame, "Allez vous reposer !",
                              0.44, 1.3, (0, 220, 255), 3)
                centered_text(frame, "Zone de risque detectee",
                              0.54, 0.7, (255, 255, 255), 2)
                centered_text(frame,
                              f"({rest_show_count}/{REST_MAX_SHOWS})  {remaining}s",
                              0.63, 0.65, (180, 180, 180), 1)
            else:
                rest_show_start = None

    # ── Alarme visuelle ────────────────────────────────────────────────────────
    if alarm_triggered:
        if both_closed and is_yawning:
            reason = "YEUX FERMES + BAILLEMENT!"
        elif both_closed:
            reason = "YEUX FERMES!"
        else:
            reason = "BAILLEMENT DETECTE!"

        centered_text(frame, f"ALARME — {reason}",
                      0.25, 0.85, (0, 0, 255), 3)
        centered_text(frame, "ARRETEZ-VOUS IMMEDIATEMENT",
                      0.35, 0.65, (0, 100, 255), 2)

        # Cadre rouge clignotant
        thicc = thicc + 2 if thicc < 16 else 2
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), thicc)
    else:
        thicc = 2

    cv2.imshow('DDD — Detection Somnolence', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Application fermee.")