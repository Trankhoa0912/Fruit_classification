from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
import io
import base64

import matplotlib

# Phải đặt trước khi import pyplot
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI()


# ============================================================
# CORS
# Cho phép frontend Live Server port 5500 gọi FastAPI port 8000
# ============================================================

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


# ============================================================
# TEMPLATE
# ============================================================

templates = Jinja2Templates(directory="templates")


# ============================================================
# COLOR HISTOGRAM
# ============================================================

def generate_color_histogram(image_bgr, bins):
    """
    Tạo biểu đồ Color Histogram cho 3 kênh:
    Blue - Green - Red.
    """

    # Đảm bảo bins hợp lệ
    bins = max(1, min(int(bins), 256))

    colors = ("b", "g", "r")

    plt.figure(figsize=(5, 3))

    for i, color in enumerate(colors):
        hist = cv2.calcHist(
            [image_bgr],
            [i],
            None,
            [bins],
            [0, 256]
        )

        plt.plot(
            hist,
            color=color,
            linewidth=1.5
        )

    # Trục X phải theo giá trị pixel 0-256
    # chứ không phải số bins
    plt.xlim([0, 256])

    plt.title(
        "Color Histogram",
        fontsize=10
    )

    plt.xlabel(
        "Pixel Intensity",
        fontsize=8
    )

    plt.ylabel(
        "Number of Pixels",
        fontsize=8
    )

    plt.tight_layout()

    # Lưu biểu đồ vào RAM
    buf = io.BytesIO()

    plt.savefig(
        buf,
        format="png",
        dpi=100
    )

    buf.seek(0)

    # Chuyển thành Base64
    base64_string = base64.b64encode(
        buf.getvalue()
    ).decode("utf-8")

    # Giải phóng matplotlib
    plt.close()

    buf.close()

    return base64_string


# ============================================================
# HÀM CHUYỂN ẢNH OPENCV -> BASE64
# ============================================================

def image_to_base64(image, ext=".jpg"):
    """
    Chuyển ảnh OpenCV thành Base64
    để gửi về frontend.
    """

    success, buffer = cv2.imencode(
        ext,
        image
    )

    if not success:
        return None

    return base64.b64encode(
        buffer
    ).decode("utf-8")


# ============================================================
# TÍNH TỶ LỆ PIXEL THEO KHOẢNG HSV
# ============================================================

def hsv_ratio(
    hsv_roi,
    object_mask,
    lower,
    upper
):
    """
    Tính tỷ lệ pixel của object nằm trong
    một khoảng HSV nhất định.
    """

    color_mask = cv2.inRange(
        hsv_roi,
        np.array(
            lower,
            dtype=np.uint8
        ),
        np.array(
            upper,
            dtype=np.uint8
        )
    )

    # Chỉ giữ pixel nằm trong object
    color_mask = cv2.bitwise_and(
        color_mask,
        color_mask,
        mask=object_mask
    )

    total_pixels = cv2.countNonZero(
        object_mask
    )

    if total_pixels == 0:
        return 0.0

    color_pixels = cv2.countNonZero(
        color_mask
    )

    return (
        color_pixels
        /
        total_pixels
    )


# ============================================================
# PHÂN LOẠI TRÁI CÂY BẰNG HSV
# ============================================================

def classify_fruit_hsv(
    hsv_roi,
    object_mask
):
    """
    Phân loại trái cây dựa trên màu HSV.

    OpenCV HSV:
        H: 0 - 179
        S: 0 - 255
        V: 0 - 255
    """

    # ========================================================
    # RED
    #
    # Red nằm ở hai đầu Hue trong OpenCV
    # ========================================================

    red1 = hsv_ratio(
        hsv_roi,
        object_mask,
        (0, 80, 50),
        (10, 255, 255)
    )

    red2 = hsv_ratio(
        hsv_roi,
        object_mask,
        (170, 80, 50),
        (179, 255, 255)
    )

    red = red1 + red2


    # ========================================================
    # ORANGE
    # ========================================================

    orange = hsv_ratio(
        hsv_roi,
        object_mask,
        (10, 100, 70),
        (22, 255, 255)
    )


    # ========================================================
    # YELLOW
    # ========================================================

    yellow = hsv_ratio(
        hsv_roi,
        object_mask,
        (22, 80, 80),
        (35, 255, 255)
    )


    # ========================================================
    # GREEN
    # ========================================================

    green = hsv_ratio(
        hsv_roi,
        object_mask,
        (35, 50, 40),
        (85, 255, 255)
    )


    # ========================================================
    # PURPLE
    # ========================================================

    purple = hsv_ratio(
        hsv_roi,
        object_mask,
        (125, 40, 40),
        (169, 255, 255)
    )


    # ========================================================
    # PIXEL CỦA OBJECT
    # ========================================================

    pixels = hsv_roi[
        object_mask > 0
    ]

    if len(pixels) == 0:
        return "Unknown", 0.0


    # Brightness trung bình
    mean_v = float(
        np.mean(
            pixels[:, 2]
        )
    )


    # ========================================================
    # RULE PHÂN LOẠI
    # ========================================================

    # Apple đỏ
    if red > 0.35:
        return "Apple", red


    # Orange
    elif orange > 0.35:
        return "Orange", orange


    # Grape
    elif purple > 0.30:
        return "Grape", purple


    # Banana / Lemon
    elif yellow > 0.30:

        # Lemon thường sáng hơn
        if mean_v > 195:
            return "Lemon", yellow

        return "Banana", yellow


    # Apple xanh
    elif green > 0.35:
        return "Green Apple", green


    # Không chắc
    max_ratio = max(
        red,
        orange,
        yellow,
        green,
        purple
    )

    return "Unknown", max_ratio


# ============================================================
# XỬ LÝ THỐNG KÊ + HSV
# ============================================================

def process_statistical(
    image_bgr,
    sample_bbox=None,
    k_factor=1.8,
    min_area=2000
):
    """
    Xử lý ảnh bằng HSV:
    - lấy vùng mẫu
    - tính Mean / Std HSV
    - tạo HSV mask
    - morphology
    - contour
    - nhận diện fruit
    - statistics
    """

    h_s, w_s = image_bgr.shape[:2]


    # ========================================================
    # ROI SAMPLE
    # ========================================================

    if sample_bbox:

        x, y, w, h = sample_bbox

    else:

        x = int(
            w_s * 0.25
        )

        y = int(
            h_s * 0.25
        )

        w = int(
            w_s * 0.5
        )

        h = int(
            h_s * 0.5
        )


    # ========================================================
    # CONVERT ROI -> HSV
    # ========================================================

    roi = image_bgr[
        y:y + h,
        x:x + w
    ]

    roi_hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )


    # ========================================================
    # LỌC PIXEL HỢP LỆ
    # ========================================================

    valid_pixels = roi_hsv[
        (roi_hsv[:, :, 1] > 40)
        &
        (roi_hsv[:, :, 2] > 40)
    ]


    # ========================================================
    # MEAN HSV
    # ========================================================

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

        flattened_roi = roi_hsv.reshape(
            -1,
            3
        )

        mean_hsv = np.mean(
            flattened_roi,
            axis=0
        )

        std_hsv = np.std(
            flattened_roi,
            axis=0
        )


    # Tránh std quá nhỏ
    std_hsv = np.maximum(
        std_hsv,
        [5.0, 20.0, 20.0]
    )


    # ========================================================
    # IMAGE -> HSV
    # ========================================================

    hsv_img = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV
    )


    # ========================================================
    # HSV RANGE
    #
    # k_factor do người dùng chỉnh trên frontend
    # ========================================================

    lower = np.clip(
        mean_hsv
        -
        k_factor * std_hsv,
        [0, 50, 50],
        [179, 255, 255]
    ).astype(
        np.uint8
    )


    upper = np.clip(
        mean_hsv
        +
        k_factor * std_hsv,
        [0, 50, 50],
        [179, 255, 255]
    ).astype(
        np.uint8
    )


    # ========================================================
    # HSV MASK
    # ========================================================

    mask = cv2.inRange(
        hsv_img,
        lower,
        upper
    )


    # ========================================================
    # MORPHOLOGY
    # ========================================================

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )


    # OPEN
    mask_clean = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )


    # CLOSE
    mask_clean = cv2.morphologyEx(
        mask_clean,
        cv2.MORPH_CLOSE,
        kernel
    )


    # ========================================================
    # CONTOURS
    # ========================================================

    contours, _ = cv2.findContours(
        mask_clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    result_img = image_bgr.copy()

    count = 0

    total_area = 0

    detected_fruits = []


    # ========================================================
    # DUYỆT OBJECT
    # ========================================================

    for cnt in contours:

        area = cv2.contourArea(
            cnt
        )


        # Bỏ object quá nhỏ
        if area < min_area:
            continue


        count += 1

        total_area += area


        bx, by, bw, bh = cv2.boundingRect(
            cnt
        )


        # ====================================================
        # OBJECT ROI
        # ====================================================

        object_bgr = image_bgr[
            by:by + bh,
            bx:bx + bw
        ]


        # Tránh ROI lỗi
        if object_bgr.size == 0:
            continue


        object_hsv = cv2.cvtColor(
            object_bgr,
            cv2.COLOR_BGR2HSV
        )


        # ====================================================
        # OBJECT MASK
        # ====================================================

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


        # ====================================================
        # PHÂN LOẠI HSV
        # ====================================================

        fruit_label, fruit_confidence = classify_fruit_hsv(
            object_hsv,
            object_mask
        )


        detected_fruits.append(
            {
                "label": fruit_label,

                "confidence": round(
                    float(
                        fruit_confidence
                    )
                    * 100,
                    2
                ),

                "area": int(
                    area
                ),

                "bbox": {
                    "x": int(bx),
                    "y": int(by),
                    "width": int(bw),
                    "height": int(bh)
                }
            }
        )


        # ====================================================
        # BOUNDING BOX
        # ====================================================

        cv2.rectangle(
            result_img,
            (bx, by),
            (
                bx + bw,
                by + bh
            ),
            (0, 255, 0),
            2
        )


        # ====================================================
        # LABEL
        # ====================================================

        label_text = (
            f"{fruit_label} "
            f"{fruit_confidence * 100:.1f}%"
        )


        cv2.putText(
            result_img,
            label_text,
            (
                bx,
                max(
                    20,
                    by - 6
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # ========================================================
    # STATISTICS
    # ========================================================

    gray = cv2.cvtColor(
        result_img,
        cv2.COLOR_BGR2GRAY
    )


    hist = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )


    hist_sum = float(
        hist.sum()
    )


    hist_norm = (
        hist
        /
        (
            hist_sum
            +
            1e-7
        )
    )


    # ========================================================
    # ENTROPY
    # ========================================================

    valid_hist = hist_norm[
        hist_norm > 0
    ]


    entropy = -float(
        np.sum(
            valid_hist
            *
            np.log2(
                valid_hist
            )
        )
    )


    stats = {

        "mean": round(
            float(
                np.mean(
                    gray
                )
            ),
            2
        ),

        "std": round(
            float(
                np.std(
                    gray
                )
            ),
            2
        ),

        "variance": round(
            float(
                np.var(
                    gray
                )
            ),
            2
        ),

        "min": int(
            np.min(
                gray
            )
        ),

        "max": int(
            np.max(
                gray
            )
        ),

        "entropy": round(
            entropy,
            2
        )
    }


    # ========================================================
    # THÊM THÔNG TIN HSV
    # Để frontend có thể đánh giá kết quả
    # ========================================================

    hsv_info = {

        "mean_h": round(
            float(
                mean_hsv[0]
            ),
            2
        ),

        "mean_s": round(
            float(
                mean_hsv[1]
            ),
            2
        ),

        "mean_v": round(
            float(
                mean_hsv[2]
            ),
            2
        ),

        "std_h": round(
            float(
                std_hsv[0]
            ),
            2
        ),

        "std_s": round(
            float(
                std_hsv[1]
            ),
            2
        ),

        "std_v": round(
            float(
                std_hsv[2]
            ),
            2
        ),

        "lower": [
            int(v)
            for v in lower
        ],

        "upper": [
            int(v)
            for v in upper
        ]
    }


    return (
        result_img,
        mask_clean,
        count,
        int(total_area),
        stats,
        detected_fruits,
        hsv_info
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home_page(
    request: Request
):
    """
    Render giao diện web.
    """

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# ============================================================
# ANALYZE API
# ============================================================

@app.post("/analyze")
async def analyze_image(

    file: UploadFile = File(...),

    # Color Histogram
    bins: int = Form(256),

    # HSV Parameters
    k_factor: float = Form(1.8),

    min_area: int = Form(2000)

):
    """
    API nhận ảnh từ frontend.

    Trả về:
    - classification
    - confidence
    - histogram
    - statistics
    - before image
    - HSV mask
    - after image
    - HSV parameters
    """

    try:

        # ====================================================
        # VALIDATE PARAMETERS
        # ====================================================

        if bins < 1:
            bins = 1

        elif bins > 256:
            bins = 256


        if k_factor <= 0:
            k_factor = 1.8


        if min_area < 1:
            min_area = 1


        # ====================================================
        # READ IMAGE
        # ====================================================

        contents = await file.read()


        if not contents:

            return {
                "error":
                "File ảnh rỗng."
            }


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
                "error":
                "Không thể đọc hoặc xử lý ảnh."
            }


        # ====================================================
        # RESIZE
        # ====================================================

        original_height = img.shape[0]

        original_width = img.shape[1]


        new_width = 400


        new_height = max(
            1,
            int(
                new_width
                *
                original_height
                /
                original_width
            )
        )


        img_resized = cv2.resize(
            img,
            (
                new_width,
                new_height
            )
        )


        # ====================================================
        # BEFORE IMAGE
        # ====================================================

        before_image_b64 = image_to_base64(
            img_resized
        )


        # ====================================================
        # HSV PROCESSING
        # ====================================================

        (
            result_img,
            mask_clean,
            count,
            total_area,
            stats,
            detected_fruits,
            hsv_info

        ) = process_statistical(

            image_bgr=img_resized,

            sample_bbox=None,

            k_factor=k_factor,

            min_area=min_area
        )


        # ====================================================
        # COLOR HISTOGRAM
        # ====================================================

        histogram_b64 = generate_color_histogram(
            img_resized,
            bins
        )


        # ====================================================
        # MASK IMAGE
        # ====================================================

        mask_image_b64 = image_to_base64(
            mask_clean,
            ".png"
        )


        # ====================================================
        # AFTER IMAGE
        # ====================================================

        after_image_b64 = image_to_base64(
            result_img
        )


        # ====================================================
        # CLASSIFICATION
        # ====================================================

        if detected_fruits:

            best_fruit = max(
                detected_fruits,
                key=lambda x:
                    x["confidence"]
            )


            predicted_label = (
                best_fruit[
                    "label"
                ]
            )


            confidence = (
                f'{best_fruit["confidence"]:.2f}%'
            )


        else:

            predicted_label = (
                "Unknown"
            )

            confidence = "0%"


        # ====================================================
        # RESPONSE
        # ====================================================

        return {

            "filename":
                file.filename,


            # ==============================
            # Classification
            # ==============================

            "prediction":
                predicted_label,

            "confidence":
                confidence,


            # ==============================
            # Object Detection
            # ==============================

            "detected_count":
                count,

            "total_area":
                total_area,

            "objects":
                detected_fruits,


            # ==============================
            # Statistics
            # ==============================

            "stats":
                stats,


            # ==============================
            # Histogram
            # ==============================

            "histogram_b64":
                histogram_b64,


            # ==============================
            # Before / Mask / After
            # ==============================

            "before_image_b64":
                before_image_b64,

            "mask_image_b64":
                mask_image_b64,

            "after_image_b64":
                after_image_b64,


            # ==============================
            # HSV Parameters
            # ==============================

            "hsv_params": {

                "k_factor":
                    k_factor,

                "min_area":
                    min_area,

                "mean_h":
                    hsv_info[
                        "mean_h"
                    ],

                "mean_s":
                    hsv_info[
                        "mean_s"
                    ],

                "mean_v":
                    hsv_info[
                        "mean_v"
                    ],

                "std_h":
                    hsv_info[
                        "std_h"
                    ],

                "std_s":
                    hsv_info[
                        "std_s"
                    ],

                "std_v":
                    hsv_info[
                        "std_v"
                    ],

                "lower":
                    hsv_info[
                        "lower"
                    ],

                "upper":
                    hsv_info[
                        "upper"
                    ]
            }
        }


    except Exception as e:

        print(
            "ERROR:",
            str(e)
        )

        return {
            "error":
                str(e)
        }