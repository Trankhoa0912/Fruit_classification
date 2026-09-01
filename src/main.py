from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

import cv2
import numpy as np


app = FastAPI()

# Nạp thư mục chứa giao diện HTML
templates = Jinja2Templates(directory="templates")


# ==========================================================
# 1. PHÂN LOẠI TRÁI CÂY DỰA TRÊN HSV
# ==========================================================

def calculate_hsv_ratio(hsv_roi, object_mask, lower, upper):
    """
    Tính tỷ lệ pixel nằm trong một khoảng HSV nhất định.
    """

    color_mask = cv2.inRange(
        hsv_roi,
        np.array(lower, dtype=np.uint8),
        np.array(upper, dtype=np.uint8)
    )

    # Chỉ xét các pixel thuộc object
    color_mask = cv2.bitwise_and(
        color_mask,
        color_mask,
        mask=object_mask
    )

    object_pixels = cv2.countNonZero(object_mask)

    if object_pixels == 0:
        return 0.0

    color_pixels = cv2.countNonZero(color_mask)

    return color_pixels / object_pixels


def classify_fruit_hsv(hsv_roi, object_mask):
    """
    Phân loại trái cây dựa trên tỷ lệ màu HSV.

    OpenCV HSV:
        H: 0 - 179
        S: 0 - 255
        V: 0 - 255
    """

    # ------------------------------------------------------
    # ĐỎ
    # Red bị chia thành 2 vùng vì Hue quay vòng
    # ------------------------------------------------------

    red_ratio_1 = calculate_hsv_ratio(
        hsv_roi,
        object_mask,
        (0, 80, 50),
        (10, 255, 255)
    )

    red_ratio_2 = calculate_hsv_ratio(
        hsv_roi,
        object_mask,
        (170, 80, 50),
        (179, 255, 255)
    )

    red_ratio = red_ratio_1 + red_ratio_2

    # ------------------------------------------------------
    # CAM
    # ------------------------------------------------------

    orange_ratio = calculate_hsv_ratio(
        hsv_roi,
        object_mask,
        (10, 100, 70),
        (22, 255, 255)
    )

    # ------------------------------------------------------
    # VÀNG
    # ------------------------------------------------------

    yellow_ratio = calculate_hsv_ratio(
        hsv_roi,
        object_mask,
        (22, 80, 80),
        (35, 255, 255)
    )

    # ------------------------------------------------------
    # XANH LÁ
    # ------------------------------------------------------

    green_ratio = calculate_hsv_ratio(
        hsv_roi,
        object_mask,
        (35, 50, 40),
        (85, 255, 255)
    )

    # ------------------------------------------------------
    # TÍM
    # ------------------------------------------------------

    purple_ratio = calculate_hsv_ratio(
        hsv_roi,
        object_mask,
        (125, 40, 40),
        (169, 255, 255)
    )

    # ------------------------------------------------------
    # Tính brightness trung bình
    # ------------------------------------------------------

    pixels = hsv_roi[object_mask > 0]

    if len(pixels) == 0:
        return "Unknown", 0.0, {}

    mean_h = float(np.mean(pixels[:, 0]))
    mean_s = float(np.mean(pixels[:, 1]))
    mean_v = float(np.mean(pixels[:, 2]))

    # ------------------------------------------------------
    # SCORE CHO TỪNG LOẠI TRÁI CÂY
    # ------------------------------------------------------

    scores = {
        "Apple": max(red_ratio, green_ratio),
        "Orange": orange_ratio,
        "Banana": yellow_ratio,
        "Lemon": yellow_ratio,
        "Grape": purple_ratio
    }

    # ------------------------------------------------------
    # Banana và Lemon đều vàng
    # Dùng brightness để tách sơ bộ
    #
    # Lemon thường sáng hơn.
    # ------------------------------------------------------

    if yellow_ratio > 0.30:

        if mean_v > 190:
            scores["Lemon"] += 0.15
        else:
            scores["Banana"] += 0.15

    # ------------------------------------------------------
    # Chọn score cao nhất
    # ------------------------------------------------------

    predicted_fruit = max(scores, key=scores.get)
    confidence = scores[predicted_fruit]

    # Nếu màu không rõ → Unknown
    if confidence < 0.15:
        predicted_fruit = "Unknown"

    details = {
        "red": round(red_ratio, 3),
        "orange": round(orange_ratio, 3),
        "yellow": round(yellow_ratio, 3),
        "green": round(green_ratio, 3),
        "purple": round(purple_ratio, 3),

        "mean_h": round(mean_h, 2),
        "mean_s": round(mean_s, 2),
        "mean_v": round(mean_v, 2)
    }

    return predicted_fruit, confidence, details


# ==========================================================
# 2. XỬ LÝ ẢNH THỐNG KÊ + DETECT OBJECT + HSV
# ==========================================================

def process_statistical(
    image_bgr,
    sample_bbox=None,
    k_factor=1.8,
    min_area=2000
):

    h_s, w_s = image_bgr.shape[:2]

    # ------------------------------------------------------
    # ROI dùng để lấy mẫu màu
    # ------------------------------------------------------

    if sample_bbox:
        x, y, w, h = sample_bbox
    else:
        x = int(w_s * 0.25)
        y = int(h_s * 0.25)
        w = int(w_s * 0.5)
        h = int(h_s * 0.5)

    roi = image_bgr[y:y+h, x:x+w]

    roi_hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    # Bỏ pixel quá tối / ít màu
    valid_pixels = roi_hsv[
        (roi_hsv[:, :, 1] > 40) &
        (roi_hsv[:, :, 2] > 40)
    ]

    # ------------------------------------------------------
    # Mean + Standard deviation HSV
    # ------------------------------------------------------

    if len(valid_pixels) >= 30:

        mean_hsv = np.mean(
            valid_pixels,
            axis=0
        )

        std_hsv = np.std(
            valid_pixels,
            axis=0
        )

    else:

        flat_roi = roi_hsv.reshape(-1, 3)

        mean_hsv = np.mean(
            flat_roi,
            axis=0
        )

        std_hsv = np.std(
            flat_roi,
            axis=0
        )

    std_hsv = np.maximum(
        std_hsv,
        [5.0, 20.0, 20.0]
    )

    # ------------------------------------------------------
    # Convert ảnh toàn bộ sang HSV
    # ------------------------------------------------------

    hsv_img = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV
    )

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

    # ------------------------------------------------------
    # Threshold
    # ------------------------------------------------------

    mask = cv2.inRange(
        hsv_img,
        lower,
        upper
    )

    # ------------------------------------------------------
    # Morphology Cleaning
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Tìm contour
    # ------------------------------------------------------

    contours, _ = cv2.findContours(
        mask_clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result_img = image_bgr.copy()

    count = 0
    total_area = 0

    detected_fruits = []

    # ======================================================
    # DUYỆT TỪNG OBJECT
    # ======================================================

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < min_area:
            continue

        count += 1
        total_area += area

        bx, by, bw, bh = cv2.boundingRect(cnt)

        # --------------------------------------------------
        # ROI của object
        # --------------------------------------------------

        object_bgr = image_bgr[
            by:by+bh,
            bx:bx+bw
        ]

        object_hsv = cv2.cvtColor(
            object_bgr,
            cv2.COLOR_BGR2HSV
        )

        # --------------------------------------------------
        # Tạo mask chính xác theo contour
        # --------------------------------------------------

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

        # --------------------------------------------------
        # PHÂN LOẠI HSV
        # --------------------------------------------------

        fruit_label, confidence, color_details = classify_fruit_hsv(
            object_hsv,
            object_mask
        )

        detected_fruits.append({
            "id": count,
            "label": fruit_label,
            "confidence": round(float(confidence) * 100, 2),
            "area": int(area),
            "bbox": {
                "x": int(bx),
                "y": int(by),
                "width": int(bw),
                "height": int(bh)
            },
            "hsv": color_details
        })

        # --------------------------------------------------
        # Vẽ Bounding Box
        # --------------------------------------------------

        cv2.rectangle(
            result_img,
            (bx, by),
            (bx + bw, by + bh),
            (0, 255, 0),
            2
        )

        text = f"{fruit_label} {confidence * 100:.1f}%"

        cv2.putText(
            result_img,
            text,
            (bx, max(20, by - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # ======================================================
    # THỐNG KÊ GRAYSCALE
    # ======================================================

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

    hist_norm = hist / (
        hist.sum() + 1e-7
    )

    entropy = -float(
        np.sum(
            [
                p * np.log2(p)
                for p in hist_norm
                if p > 0
            ]
        )
    )

    stats = {
        "mean": round(
            float(np.mean(gray)),
            2
        ),

        "std": round(
            float(np.std(gray)),
            2
        ),

        "variance": round(
            float(np.var(gray)),
            2
        ),

        "min": int(
            np.min(gray)
        ),

        "max": int(
            np.max(gray)
        ),

        "entropy": round(
            entropy,
            2
        )
    }

    return (
        result_img,
        mask_clean,
        count,
        int(total_area),
        stats,
        detected_fruits
    )


# ==========================================================
# 3. TRANG CHỦ
# ==========================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home_page(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# ==========================================================
# 4. API PHÂN TÍCH ẢNH
# ==========================================================

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...)
):

    try:

        # --------------------------------------------------
        # Đọc file ảnh upload
        # --------------------------------------------------

        contents = await file.read()

        nparr = np.frombuffer(
            contents,
            np.uint8
        )

        img = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        if img is None:

            return {
                "error":
                "Không thể đọc hoặc xử lý ảnh."
            }

        # --------------------------------------------------
        # Resize
        # --------------------------------------------------

        new_width = 400

        new_height = int(
            new_width *
            img.shape[0] /
            img.shape[1]
        )

        img_resized = cv2.resize(
            img,
            (
                new_width,
                new_height
            )
        )

        # --------------------------------------------------
        # Xử lý ảnh
        # --------------------------------------------------

        (
            result_img,
            mask_clean,
            count,
            total_area,
            stats,
            detected_fruits
        ) = process_statistical(
            image_bgr=img_resized,
            sample_bbox=None,
            k_factor=1.8,
            min_area=2000
        )

        # --------------------------------------------------
        # Xác định prediction chung
        # --------------------------------------------------

        if detected_fruits:

            # Object có confidence cao nhất
            best_fruit = max(
                detected_fruits,
                key=lambda x: x["confidence"]
            )

            predicted_label = best_fruit["label"]

            confidence = (
                f'{best_fruit["confidence"]:.2f}%'
            )

        else:

            predicted_label = "Unknown"

            confidence = "0%"

        # --------------------------------------------------
        # Trả JSON về frontend
        # --------------------------------------------------

        return {

            "filename":
                file.filename,

            "prediction":
                predicted_label,

            "confidence":
                confidence,

            "detected_count":
                count,

            "total_area":
                total_area,

            "stats":
                stats,

            "objects":
                detected_fruits
        }

    except Exception as e:

        return {
            "error": str(e)
        }
