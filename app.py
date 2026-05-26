"""
Phát hiện biển báo giao thông — Gradio + YOLO (best.pt).
Chạy: python app.py
"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WEIGHTS_PT = ROOT / "best.pt"
WEIGHTS_ONNX = ROOT / "best.onnx"

DISPLAY_NAMES = {
    "Het tat ca cac lenh cam": "Hết tất cả các lệnh cấm",
    "Cam di nguoc chieu": "Cấm đi ngược chiều",
    "Cam o to": "Cấm ô tô",
    "Cam o to re phai": "Cấm ô tô rẽ phải",
    "Cam o to re trai": "Cấm ô tô rẽ trái",
    "Cam mo to": "Cấm mô tô",
    "Cam xe tai": "Cấm xe tải",
    "Cam xe tai (gioi han tai trong)": "Cấm xe tải (giới hạn tải trọng)",
    "Cam xe o to khach": "Cấm xe ô tô khách",
    "Cam nguoi di bo": "Cấm người đi bộ",
    "Han che trong luong xe": "Hạn chế trọng lượng xe",
    "Han che chieu cao xe": "Hạn chế chiều cao xe",
    "Cam re trai": "Cấm rẽ trái",
    "Cam re phai": "Cấm rẽ phải",
    "Cam quay dau xe": "Cấm quay đầu xe",
    "Cam o to quay dau xe": "Cấm ô tô quay đầu xe",
    "Cam re trai va quay dau xe": "Cấm rẽ trái và quay đầu xe",
    "Toc do toi da cho phep": "Tốc độ tối đa cho phép",
    "Cam su dung coi": "Cấm sử dụng còi",
    "Cam dung xe va do xe": "Cấm dừng xe và đỗ xe",
    "Cam do xe": "Cấm đỗ xe",
    "Cam re trai va re phai": "Cấm rẽ trái và rẽ phải",
    "Cam xe dap dien va xe may dien": "Cấm xe đạp điện và xe máy điện",
    "Huong di phai theo (di thang va re phai)": "Hướng đi phải theo (đi thẳng và rẽ phải)",
    "Huong di phai theo (re trai)": "Hướng đi phải theo (rẽ trái)",
    "Huong di phai theo (re phai)": "Hướng đi phải theo (rẽ phải)",
    "Huong phai di vong chuong ngai vat (phia phai)": "Hướng phải đi vòng chướng ngại vật (phía phải)",
    "Huong phai di vong chuong ngai vat (phia trai)": "Hướng phải đi vòng chướng ngại vật (phía trái)",
    "Noi giao nhau chay theo vong xuyen": "Nơi giao nhau chạy theo vòng xuyến",
    "Duong mot chieu": "Đường một chiều",
    "Cho quay xe": "Chỗ quay xe",
    "Bat dau duong cao toc": "Bắt đầu đường cao tốc",
    "Lan duong danh rieng cho xe buyt": "Làn đường dành riêng cho xe buýt",
    "Thuyet minh bien chinh - Cam dung va do xe": "Thuyết minh biển chính - Cấm dừng và đỗ xe",
    "Cho ngoat nguy hiem (vong trai)": "Chỗ ngoặt nguy hiểm (vòng trái)",
    "Cho ngoat nguy hiem (vong phai)": "Chỗ ngoặt nguy hiểm (vòng phải)",
    "Nhieu cho ngoat nguy hiem lien tiep (vong trai)": "Nhiều chỗ ngoặt nguy hiểm liên tiếp (vòng trái)",
    "Nhieu cho ngoat nguy hiem lien tiep (vong phai)": "Nhiều chỗ ngoặt nguy hiểm liên tiếp (vòng phải)",
    "Duong bi thu hep (hep ben trai)": "Đường bị thu hẹp (hẹp bên trái)",
    "Duong bi thu hep (hep ben phai)": "Đường bị thu hẹp (hẹp bên phải)",
    "Duong giao nhau (nga tu)": "Đường giao nhau (ngã tư)",
    "Duong giao nhau (nga ba phai)": "Đường giao nhau (ngã ba phải)",
    "Duong giao nhau (nga ba trai)": "Đường giao nhau (ngã ba trái)",
    "Giao nhau voi duong khong uu tien (hai phia)": "Giao nhau với đường không ưu tiên (hai phía)",
    "Giao nhau voi duong khong uu tien (ben phai)": "Giao nhau với đường không ưu tiên (bên phải)",
    "Giao nhau voi duong khong uu tien (ben trai)": "Giao nhau với đường không ưu tiên (bên trái)",
    "Giao nhau voi duong uu tien": "Giao nhau với đường ưu tiên",
    "Giao nhau co tin hieu den": "Giao nhau có tín hiệu đèn",
    "Giao nhau voi duong sat co rao chan": "Giao nhau với đường sắt có rào chắn",
    "Doc xuong nguy hiem": "Dốc xuống nguy hiểm",
    "Nguoi di bo cat ngang": "Người đi bộ cắt ngang",
    "Tre em": "Trẻ em",
    "Cong truong": "Công trường",
    "Nguy hiem khac": "Nguy hiểm khác",
    "Duong doi": "Đường đôi",
    "Chuong ngai vat tren duong": "Chướng ngại vật trên đường",
}


def resolve_model_path() -> Path:
    if WEIGHTS_PT.exists():
        return WEIGHTS_PT
    if WEIGHTS_ONNX.exists():
        return WEIGHTS_ONNX
    raise FileNotFoundError("Không tìm thấy best.pt hoặc best.onnx trong thư mục dự án.")


def label_vi(raw: str) -> str:
    return DISPLAY_NAMES.get(raw, raw)


def classify_warning(raw: str) -> str:
    if raw.startswith("Cam ") or raw.startswith("Han che") or "Het tat ca" in raw:
        return "Cấm / Nguy hiểm"
    if "Toc do toi da" in raw:
        return "Tốc độ"
    if any(
        k in raw
        for k in (
            "Cho ngoat", "Giao nhau", "Doc xuong", "Nguoi di bo", "Tre em",
            "Cong truong", "Nguy hiem", "Duong bi thu hep", "Chuong ngai", "Duong doi",
        )
    ):
        return "Cảnh báo"
    return "Chỉ dẫn"


def boxes_to_labels(boxes, names: dict) -> list[str]:
    if boxes is None or len(boxes) == 0:
        return []
    return [label_vi(names[int(c)]) for c in boxes.cls.int().tolist()]


# =============================
# 1. Load YOLO model
# =============================
model = YOLO(str(resolve_model_path()), task="detect")
CLASS_NAMES = model.names

# =============================
# 2. Hàm detect ảnh
# =============================
def detect_signs(files, conf_threshold):
    if not files:
        return [], [], "Chưa tải ảnh nào."

    gallery_results = []
    table_results = []
    total_signs = 0
    all_labels: list[str] = []

    for file in files:
        path = file.name if hasattr(file, "name") else str(file)
        filename = Path(path).name

        results = model(path, conf=conf_threshold, verbose=False)
        boxes = results[0].boxes
        num_signs = 0 if boxes is None else len(boxes)
        total_signs += num_signs

        annotated_img = results[0].plot()
        labels = boxes_to_labels(boxes, CLASS_NAMES)
        all_labels.extend(labels)

        if num_signs > 0:
            max_conf = float(boxes.conf.max())
            status = "Biển báo phát hiện"
            signs_text = "; ".join(sorted(set(labels)))
            raw_list = [CLASS_NAMES[int(c)] for c in boxes.cls.int().tolist()]
            warning = _worst_warning(raw_list)
        else:
            max_conf = 0.0
            status = "Không phát hiện"
            signs_text = "-"
            warning = "-"

        gallery_results.append(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        table_results.append([
            filename,
            status,
            num_signs,
            round(max_conf, 3),
            warning,
            signs_text,
        ])

    label_counts = Counter(all_labels)
    top_signs = ", ".join(f"{name} ({n})" for name, n in label_counts.most_common(5))
    if not top_signs:
        top_signs = "Không có"

    summary_text = f"""
📊 **Tổng số ảnh:** {len(files)}  
🚦 **Tổng số biển báo phát hiện:** {total_signs}  
🏷️ **Biển xuất hiện nhiều nhất:** {top_signs}
"""
    return gallery_results, table_results, summary_text


# =============================
# 3. Hàm detect video
# =============================
def detect_video(video_file, conf_threshold):
    if video_file is None:
        return None, [], "Chưa tải video."

    path = video_file if isinstance(video_file, str) else video_file.name
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, [], "Không mở được file video."

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_path = Path(tempfile.gettempdir()) / "traffic_sign_output.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    sign_counter: Counter = Counter()
    frames_with_signs = 0
    frame_idx = 0
    max_conf_video = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf_threshold, verbose=False)
        boxes = results[0].boxes
        annotated = results[0].plot()
        writer.write(annotated)

        if boxes is not None and len(boxes) > 0:
            frames_with_signs += 1
            max_conf_video = max(max_conf_video, float(boxes.conf.max()))
            for raw in [CLASS_NAMES[int(c)] for c in boxes.cls.int().tolist()]:
                sign_counter[label_vi(raw)] += 1

        frame_idx += 1

    cap.release()
    writer.release()

    table_results = [
        [name, count, _warning_for_display(name)]
        for name, count in sign_counter.most_common()
    ]

    summary_text = f"""
📊 **Tổng số frame:** {frame_idx}  
🚦 **Frame có biển báo:** {frames_with_signs}  
🏷️ **Loại biển khác nhau:** {len(sign_counter)}  
🎯 **Confidence cao nhất:** {max_conf_video:.3f}
"""
    if not table_results:
        table_results = [["-", 0, "-"]]

    return str(out_path), table_results, summary_text


def _warning_for_display(vi_name: str) -> str:
    raw = next((k for k, v in DISPLAY_NAMES.items() if v == vi_name), vi_name)
    return classify_warning(raw)


def _worst_warning(raw_names: list[str]) -> str:
    order = {"Cấm / Nguy hiểm": 0, "Cảnh báo": 1, "Tốc độ": 2, "Chỉ dẫn": 3}
    levels = [classify_warning(r) for r in raw_names]
    return min(levels, key=lambda x: order.get(x, 9))


# =============================
# 4. Giao diện Gradio
# =============================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:

    gr.Markdown(
        f"""
# 🚦 Hệ thống nhận diện biển báo giao thông Việt Nam  
"""
    )

    conf_slider = gr.Slider(
        minimum=0.1,
        maximum=0.9,
        value=0.5,
        step=0.05,
        label="🎯 Ngưỡng confidence",
    )

    with gr.Tabs():
        with gr.Tab("📷 Ảnh"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_files = gr.File(
                        file_types=["image"],
                        file_count="multiple",
                        label="📂 Tải ảnh",
                    )
                    img_btn = gr.Button("🔍 Phân tích ảnh", variant="primary")

                with gr.Column(scale=2):
                    img_gallery = gr.Gallery(
                        label="🖼 Kết quả nhận diện",
                        columns=2,
                        height="auto",
                    )

            gr.Markdown("## 📈 Tổng hợp ảnh")
            img_table = gr.Dataframe(
                headers=["Tên file", "Trạng thái", "Số biển", "Conf max", "Cảnh báo", "Biển phát hiện"],
                interactive=False,
            )
            img_summary = gr.Markdown()

            img_btn.click(
                fn=detect_signs,
                inputs=[input_files, conf_slider],
                outputs=[img_gallery, img_table, img_summary],
            )

        with gr.Tab("🎬 Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_video = gr.Video(label="📂 Tải video")
                    vid_btn = gr.Button("🔍 Phân tích video", variant="primary")

                with gr.Column(scale=2):
                    output_video = gr.Video(label="🖼 Video đã gắn nhãn")

            gr.Markdown("## 📈 Tổng hợp video")
            vid_table = gr.Dataframe(
                headers=["Biển báo", "Số lần xuất hiện", "Mức cảnh báo"],
                interactive=False,
            )
            vid_summary = gr.Markdown()

            vid_btn.click(
                fn=detect_video,
                inputs=[input_video, conf_slider],
                outputs=[output_video, vid_table, vid_summary],
            )

demo.launch(debug=True, share=True)
