from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
import io
import base64

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")


def image_to_base64(image, ext=".jpg"):
    success, buffer = cv2.imencode(ext, image)
    if not success:
        return None
    return base64.b64encode(buffer).decode("utf-8")


def generate_color_histogram(image_bgr, bins):
    bins = max(1, min(int(bins), 256))
    colors = ("b", "g", "r")
    labels = ("Blue", "Green", "Red")

    plt.figure(figsize=(5, 3))

    x = np.linspace(0, 255, bins)

    for i, color in enumerate(colors):
        hist = cv2.calcHist(
            [image_bgr],
            [i],
            None,
            [bins],
            [0, 256]
        )
        plt.plot(x, hist.ravel(), color=color, linewidth=1.5, label=labels[i])

    plt.xlim([0, 255])
    plt.title("Color Histogram", fontsize=10)
    plt.xlabel("Pixel Intensity", fontsize=8)
    plt.ylabel("Number of Pixels", fontsize=8)
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)

    base64_string = base64.b64encode(buf.getvalue()).decode("utf-8")

    plt.close()
    buf.close()

    return base64_string


def hsv_ratio(hsv_roi, object_mask, lower, upper):
    color_mask = cv2.inRange(
        hsv_roi,
        np.array(lower, dtype=np.uint8),
        np.array(upper, dtype=np.uint8),
    )

    color_mask = cv2.bitwise_and(
        color_mask,
        color_mask,
        mask=object_mask
    )

    total_pixels = cv2.countNonZero(object_mask)

    if total_pixels == 0:
        return 0.0

    color_pixels = cv2.countNonZero(color_mask)
    return color_pixels / total_pixels


def classify_fruit_hsv(hsv_roi, object_mask):
    red1 = hsv_ratio(
        hsv_roi, object_mask,
        (0, 80, 50),
        (10, 255, 255)
    )

    red2 = hsv_ratio(
        hsv_roi, object_mask,
        (170, 80, 50),
        (179, 255, 255)
    )

    red = red1 + red2

    orange = hsv_ratio(
        hsv_roi, object_mask,
        (10, 100, 70),
        (22, 255, 255)
    )

    yellow = hsv_ratio(
        hsv_roi, object_mask,
        (22, 80, 80),
        (35, 255, 255)
    )

    green = hsv_ratio(
        hsv_roi, object_mask,
        (35, 50, 40),
        (85, 255, 255)
    )

    purple = hsv_ratio(
        hsv_roi, object_mask,
        (125, 40, 40),
        (169, 255, 255)
    )

    pixels = hsv_roi[object_mask > 0]

    if len(pixels) == 0:
        return "Unknown", 0.0

    mean_v = float(np.mean(pixels[:, 2]))

    if red > 0.35:
        return "Apple", red

    if orange > 0.35:
        return "Orange", orange

    if purple > 0.30:
        return "Grape", purple

    if yellow > 0.30:
        if mean_v > 195:
            return "Lemon", yellow
        return "Banana", yellow

    if green > 0.35:
        return "Green Apple", green

    max_ratio = max(red, orange, yellow, green, purple)
    return "Unknown", max_ratio


def process_statistical(
    image_bgr,
    sample_bbox=None,
    k_factor=1.8,
    min_area=2000,
    stat_k=0.0,
):
    h_s, w_s = image_bgr.shape[:2]

    if sample_bbox:
        x, y, w, h = sample_bbox
    else:
        x = int(w_s * 0.25)
        y = int(h_s * 0.25)
        w = int(w_s * 0.50)
        h = int(h_s * 0.50)

    roi = image_bgr[y:y + h, x:x + w]

    if roi.size == 0:
        raise ValueError("ROI mẫu không hợp lệ.")

    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    valid_pixels = roi_hsv[
        (roi_hsv[:, :, 1] > 40) &
        (roi_hsv[:, :, 2] > 40)
    ]

    if len(valid_pixels) >= 30:
        mean_hsv = np.mean(valid_pixels, axis=0)
        std_hsv = np.std(valid_pixels, axis=0)
    else:
        flattened_roi = roi_hsv.reshape(-1, 3)
        mean_hsv = np.mean(flattened_roi, axis=0)
        std_hsv = np.std(flattened_roi, axis=0)

    std_hsv = np.maximum(
        std_hsv,
        np.array([5.0, 20.0, 20.0])
    )

    # Ảnh HSV để xử lý
    hsv_img = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV
    )

    # Nếu muốn ảnh HSV "ảo màu" như hình mẫu,
    # chỉ cần giữ raw HSV matrix này để encode ra ảnh.
    # Browser/viewer sẽ hiển thị 3 kênh đó như ảnh màu thường.
    hsv_visual = hsv_img.copy()

    lower = np.clip(
        mean_hsv - k_factor * std_hsv,
        [0, 50, 50],
        [179, 255, 255]
    ).astype(np.uint8)

    upper = np.clip(
        mean_hsv + k_factor * std_hsv,
        [0, 50, 50],
        [179, 255, 255]
    ).astype(np.uint8)

    mask = cv2.inRange(
        hsv_img,
        lower,
        upper
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    mask_clean = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask_clean = cv2.morphologyEx(
        mask_clean,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask_clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    count = 0
    total_area = 0
    detected_fruits = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < min_area:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)

        object_bgr = image_bgr[
            by:by + bh,
            bx:bx + bw
        ]

        if object_bgr.size == 0:
            continue

        object_hsv = cv2.cvtColor(
            object_bgr,
            cv2.COLOR_BGR2HSV
        )

        object_mask = np.zeros(
            (bh, bw),
            dtype=np.uint8
        )

        shifted_cnt = cnt.copy()
        shifted_cnt[:, :, 0] -= bx
        shifted_cnt[:, :, 1] -= by

        cv2.drawContours(
            object_mask,
            [shifted_cnt],
            -1,
            255,
            thickness=-1
        )

        fruit_label, fruit_confidence = classify_fruit_hsv(
            object_hsv,
            object_mask
        )

        count += 1
        total_area += area

        detected_fruits.append(
            {
                "label": fruit_label,
                "confidence": round(
                    float(fruit_confidence) * 100,
                    2
                ),
                "area": int(area),
                "bbox": {
                    "x": int(bx),
                    "y": int(by),
                    "width": int(bw),
                    "height": int(bh),
                },
            }
        )

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    hist = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    hist_sum = float(hist.sum())
    hist_norm = hist / (hist_sum + 1e-7)
    valid_hist = hist_norm[hist_norm > 0]

    entropy = -float(
        np.sum(valid_hist * np.log2(valid_hist))
    )

    mean_val = round(float(np.mean(gray)), 2)
    std_val = round(float(np.std(gray)), 2)
    t_val = float(np.clip(mean_val + stat_k * std_val, 0, 255))
    stats = {
        "mean": round(float(np.mean(gray)), 2),
        "std": round(float(np.std(gray)), 2),
        "variance": round(float(np.var(gray)), 2),
        "min": int(np.min(gray)),
        "max": int(np.max(gray)),
        "entropy": round(entropy, 2),
        "threshold": round(t_val, 2),
    }
    _, binary_stat = cv2.threshold(gray, t_val, 255, cv2.THRESH_BINARY)
    stat_gray_b64 = image_to_base64(gray, ".png")
    stat_bin_b64 = image_to_base64(binary_stat, ".png")

    hsv_info = {
        "mean_h": round(float(mean_hsv[0]), 2),
        "mean_s": round(float(mean_hsv[1]), 2),
        "mean_v": round(float(mean_hsv[2]), 2),
        "std_h": round(float(std_hsv[0]), 2),
        "std_s": round(float(std_hsv[1]), 2),
        "std_v": round(float(std_hsv[2]), 2),
        "lower": [int(v) for v in lower],
        "upper": [int(v) for v in upper],
    }

    return (
        hsv_visual,
        count,
        int(total_area),
        stats,
        detected_fruits,
        hsv_info,
        stat_gray_b64,
        stat_bin_b64,
    )


@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    bins: int = Form(256),
    k_factor: float = Form(1.8),
    min_area: int = Form(2000),
    stat_k: float = Form(0.0),
):
    try:
        bins = max(1, min(int(bins), 256))

        if k_factor <= 0:
            k_factor = 1.8

        if min_area < 1:
            min_area = 1

        contents = await file.read()

        if not contents:
            return {"error": "File ảnh rỗng."}

        nparr = np.frombuffer(
            contents,
            dtype=np.uint8
        )

        img = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        if img is None:
            return {
                "error": "Không thể đọc hoặc xử lý ảnh."
            }

        original_height, original_width = img.shape[:2]

        new_width = 400
        new_height = max(
            1,
            int(new_width * original_height / original_width)
        )

        img_resized = cv2.resize(
            img,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        before_image_b64 = image_to_base64(
            img_resized,
            ".jpg"
        )

        (
            hsv_visual,
            count,
            total_area,
            stats,
            detected_fruits,
            hsv_info,
            stat_gray_b64,
            stat_bin_b64,
        ) = process_statistical(
            image_bgr=img_resized,
            sample_bbox=None,
            k_factor=k_factor,
            min_area=min_area,
            stat_k=stat_k,
        )

        # Ảnh HSV kiểu "ảo màu" như ví dụ người dùng mong muốn
        hsv_image_b64 = image_to_base64(
            hsv_visual,
            ".jpg"
        )

        histogram_b64 = generate_color_histogram(
            img_resized,
            bins
        )

        if detected_fruits:
            best_fruit = max(
                detected_fruits,
                key=lambda x: x["confidence"]
            )
            predicted_label = best_fruit["label"]
            confidence = f'{best_fruit["confidence"]:.2f}%'
        else:
            predicted_label = "Unknown"
            confidence = "0%"

        return {
            "filename": file.filename,

            "prediction": predicted_label,
            "confidence": confidence,

            "detected_count": count,
            "total_area": total_area,
            "objects": detected_fruits,

            "stats": stats,
            "histogram_b64": histogram_b64,

            "before_image_b64": before_image_b64,
            "hsv_image_b64": hsv_image_b64,
            "stat_gray_b64": stat_gray_b64,
            "stat_bin_b64": stat_bin_b64,

            "hsv_params": {
                "k_factor": k_factor,
                "min_area": min_area,

                "mean_h": hsv_info["mean_h"],
                "mean_s": hsv_info["mean_s"],
                "mean_v": hsv_info["mean_v"],

                "std_h": hsv_info["std_h"],
                "std_s": hsv_info["std_s"],
                "std_v": hsv_info["std_v"],

                "lower": hsv_info["lower"],
                "upper": hsv_info["upper"],
            },
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}
