# app.py
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import cv2

from ultralytics import YOLO


# UI
st.set_page_config(page_title="Unattended baggage (YOLO11s COCO)", layout="centered")
st.title("Забытый багаж (предобученная YOLO11s на COCO)")

conf = st.slider("Порог уверенности (conf)", 0.05, 0.95, 0.25, 0.05)
norm_dist_thr = st.slider(
    "Порог 'далеко от человека' (норм. расстояние)",
    0.5, 6.0, 2.0, 0.1,
    help="distance / (sqrt(area_person) + sqrt(area_bag))"
)
show_debug = st.checkbox("Показывать отладочную таблицу (дистанции/ближайший person)", value=True)


# Model (YOLO11s COCO)
@st.cache_resource
def load_model():
    # Предобученные веса COCO: yolo11s.pt
    return YOLO("yolo11s.pt")

model = load_model()

# Находим id нужных COCO-классов по именам (устойчиво, без хардкода индексов)
NAME2ID = {v: k for k, v in model.names.items()}
PERSON_ID = NAME2ID.get("person", None)
BAG_NAMES = ["backpack", "handbag", "suitcase"]
BAG_IDS = [NAME2ID[n] for n in BAG_NAMES if n in NAME2ID]

if PERSON_ID is None or len(BAG_IDS) == 0:
    st.error("В модели не найдены классы 'person' и/или багаж (backpack/handbag/suitcase).")
    st.stop()

# Input: upload OR camera
if "mode" not in st.session_state:
    st.session_state.mode = "upload"
if "captured_image" not in st.session_state:
    st.session_state.captured_image = None
if "use_camera_preview" not in st.session_state:
    st.session_state.use_camera_preview = False

c1, c2 = st.columns(2)
with c1:
    if st.button("Загрузить фото", use_container_width=True):
        st.session_state.mode = "upload"
        st.session_state.captured_image = None
        st.session_state.use_camera_preview = False
with c2:
    if st.button("Камера", use_container_width=True):
        st.session_state.mode = "camera"
        st.session_state.captured_image = None
        st.session_state.use_camera_preview = True

selected_image = None

if st.session_state.mode == "upload":
    uploaded = st.file_uploader("Выберите файл", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is not None:
        selected_image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

elif st.session_state.mode == "camera":
    st.caption("Нажмите «Снимок», чтобы зафиксировать кадр для анализа.")
    cam = st.camera_input("Поток с камеры", disabled=not st.session_state.use_camera_preview)
    if cam is not None:
        if st.button("Снимок", use_container_width=True):
            selected_image = Image.open(cam).convert("RGB")
            st.session_state.captured_image = selected_image
            st.session_state.use_camera_preview = False
    if st.session_state.captured_image is not None:
        selected_image = st.session_state.captured_image

if selected_image is not None:
    st.image(selected_image, caption="Предпросмотр", use_container_width=True)


# Helper: unattended rule
def box_center_xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def box_area_xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def norm_distance(bag_xyxy, person_xyxy):
    # distance / (sqrt(area_person)+sqrt(area_bag))
    bx, by = box_center_xyxy(bag_xyxy)
    px, py = box_center_xyxy(person_xyxy)
    d = ((bx - px) ** 2 + (by - py) ** 2) ** 0.5
    sp = (box_area_xyxy(person_xyxy) ** 0.5)
    sb = (box_area_xyxy(bag_xyxy) ** 0.5)
    denom = max(1e-6, sp + sb)
    return d / denom

def draw_unattended(image_rgb: np.ndarray, unattended_rows: list):
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    for row in unattended_rows:
        x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
        label = f'{row["bag_type"]} {row["conf"]:.2f}'
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_bgr, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# Run analysis
btn_disabled = selected_image is None
if st.button("Запустить анализ", type="primary", use_container_width=True, disabled=btn_disabled):
    # Фильтруем по классам для скорости: person + bag classes
    classes_filter = [PERSON_ID] + BAG_IDS

    results = model.predict(
        source=selected_image,
        conf=float(conf),
        imgsz=640,
        classes=classes_filter,
        verbose=False
    )

    r0 = results[0]
    boxes = r0.boxes

    if boxes is None or len(boxes) == 0:
        st.info("Ничего не найдено.")
        st.stop()

    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()  # x1,y1,x2,y2

    # Отделяем person и bag
    persons = []
    bags = []
    for i in range(len(cls_ids)):
        row = {
            "i": i,
            "cls": int(cls_ids[i]),
            "name": r0.names.get(int(cls_ids[i]), str(int(cls_ids[i]))),
            "conf": float(confs[i]),
            "x1": float(xyxy[i][0]),
            "y1": float(xyxy[i][1]),
            "x2": float(xyxy[i][2]),
            "y2": float(xyxy[i][3]),
        }
        if row["cls"] == PERSON_ID:
            persons.append(row)
        elif row["cls"] in BAG_IDS:
            bags.append(row)

    if len(bags) == 0:
        st.info("Багаж (backpack/handbag/suitcase) не найден.")
        st.stop()

    # Правило "забытый багаж": далеко от ближайшего person с учётом размеров
    unattended = []
    debug_rows = []

    person_xyxys = [np.array([p["x1"], p["y1"], p["x2"], p["y2"]], dtype=float) for p in persons]

    for b in bags:
        b_xyxy = np.array([b["x1"], b["y1"], b["x2"], b["y2"]], dtype=float)

        if len(persons) == 0:
            min_nd = float("inf")
            nearest_idx = None
            is_unattended = True
        else:
            nds = [norm_distance(b_xyxy, p_xyxy) for p_xyxy in person_xyxys]
            nearest_idx = int(np.argmin(nds))
            min_nd = float(nds[nearest_idx])
            is_unattended = (min_nd > float(norm_dist_thr))

        debug_rows.append({
            "bag_type": b["name"],
            "bag_conf": b["conf"],
            "min_norm_dist": min_nd if np.isfinite(min_nd) else None,
            "nearest_person": nearest_idx,
            "is_unattended": is_unattended,
        })

        if is_unattended:
            unattended.append({
                "bag_type": b["name"],
                "conf": b["conf"],
                "x1": b["x1"], "y1": b["y1"], "x2": b["x2"], "y2": b["y2"],
                "min_norm_dist": min_nd,
            })

    if len(unattended) == 0:
        st.success("Забытого багажа не обнаружено (по текущему порогу).")
        if show_debug:
            st.subheader("Отладка")
            st.dataframe(pd.DataFrame(debug_rows), use_container_width=True)
        st.stop()

    # Рисуем ТОЛЬКО забытый багаж
    img_rgb = np.array(selected_image, dtype=np.uint8)
    annotated = draw_unattended(img_rgb, unattended)

    st.image(annotated, caption="BBox только забытого багажа", use_container_width=True)

    st.subheader("Забытый багаж (список)")
    df_un = pd.DataFrame(unattended).sort_values(["conf"], ascending=False)
    st.dataframe(df_un, use_container_width=True)

    if show_debug:
        st.subheader("Отладка")
        st.dataframe(pd.DataFrame(debug_rows), use_container_width=True)
