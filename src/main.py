from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg') # Cần thiết để vẽ biểu đồ ngầm trên server không có màn hình
import matplotlib.pyplot as plt
from fastapi import Form # Dùng để nhận tham số từ HTML form
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

app = FastAPI()
# Cấp quyền cho cổng 5500 của Live Server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Nạp thư mục chứa giao diện HTML
#templates = Jinja2Templates(directory="templates")

# Lấy đường dẫn thư mục templates nằm cùng cấp với file main.py trong src
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Color Histogram
def generate_color_histogram(image_bgr, bins):
    # Định nghĩa màu cho 3 kênh (Blue, Green, Red) của OpenCV
    colors = ('b', 'g', 'r')
    plt.figure(figsize=(5, 3))
    
    # Tính và vẽ biểu đồ cho từng kênh màu
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image_bgr], [i], None, [bins], [0, 256])
        plt.plot(hist, color=color, linewidth=1.5)
        plt.xlim([0, bins])
        
    plt.title('Color Histogram', fontsize=10)
    plt.xlabel('Bins', fontsize=8)
    plt.ylabel('Số lượng Pixels', fontsize=8)
    plt.tight_layout()

    # Lưu biểu đồ vào bộ nhớ đệm thay vì lưu ra ổ cứng
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    
    # Mã hóa biểu đồ thành chuỗi Base64
    base64_string = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close() # Đóng biểu đồ để giải phóng bộ nhớ
    
    return base64_string


def hsv_ratio(hsv_roi, object_mask, lower, upper):

    color_mask = cv2.inRange(
        hsv_roi,
        np.array(lower, dtype=np.uint8),
        np.array(upper, dtype=np.uint8)
    )

    # Chỉ tính những pixel nằm bên trong object
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

    # RED
    # Màu đỏ trong HSV OpenCV nằm ở 2 đầu Hue
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

    # ORANGE
    orange = hsv_ratio(
        hsv_roi,
        object_mask,
        (10, 100, 70),
        (22, 255, 255)
    )

    # YELLOW
    yellow = hsv_ratio(
        hsv_roi,
        object_mask,
        (22, 80, 80),
        (35, 255, 255)
    )

    # GREEN
    green = hsv_ratio(
        hsv_roi,
        object_mask,
        (35, 50, 40),
        (85, 255, 255)
    )

    # PURPLE
    purple = hsv_ratio(
        hsv_roi,
        object_mask,
        (125, 40, 40),
        (169, 255, 255)
    )

    # Lấy các pixel bên trong object
    pixels = hsv_roi[object_mask > 0]

    if len(pixels) == 0:
        return "Unknown", 0.0

    # Độ sáng trung bình
    mean_v = np.mean(pixels[:, 2])

    # =====================================================
    # RULE PHÂN LOẠI
    # =====================================================

    if red > 0.35:
        return "Apple", red

    elif orange > 0.35:
        return "Orange", orange

    elif purple > 0.30:
        return "Grape", purple

    elif yellow > 0.30:

        if mean_v > 195:
            return "Lemon", yellow
        else:
            return "Banana", yellow

    elif green > 0.35:
        return "Green Apple", green

    else:
        max_ratio = max(
            red,
            orange,
            yellow,
            green,
            purple
        )

        return "Unknown", max_ratio

def process_statistical(
    image_bgr,
    sample_bbox=None,
    k_factor=1.8,
    min_area=2000
):

    h_s, w_s = image_bgr.shape[:2]

    x, y, w, h = sample_bbox if sample_bbox else (
        int(w_s * 0.25),
        int(h_s * 0.25),
        int(w_s * 0.5),
        int(h_s * 0.5)
    )

    roi_hsv = cv2.cvtColor(
        image_bgr[y:y+h, x:x+w],
        cv2.COLOR_BGR2HSV
    )

    valid_pixels = roi_hsv[
        (roi_hsv[:, :, 1] > 40)
        &
        (roi_hsv[:, :, 2] > 40)
    ]

    mean_hsv = (
        np.mean(valid_pixels, axis=0)
        if len(valid_pixels) >= 30
        else np.mean(roi_hsv.reshape(-1, 3), axis=0)
    )

    std_hsv = np.maximum(
        np.std(valid_pixels, axis=0)
        if len(valid_pixels) >= 30
        else np.std(roi_hsv.reshape(-1, 3), axis=0),
        [5.0, 20.0, 20.0]
    )

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
        cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        ),
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask_clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result_img = image_bgr.copy()

    count, total_area = 0, 0

    # =====================================================
    # Danh sách chứa kết quả phân loại object
    # =====================================================
    detected_fruits = []


    # =====================================================
    # CODE CONTOUR CŨ + THÊM PHÂN LOẠI HSV
    # =====================================================
    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area >= min_area:

            count += 1
            total_area += area

            bx, by, bw, bh = cv2.boundingRect(cnt)


            # =================================================
            # LẤY OBJECT ĐỂ PHÂN LOẠI HSV
            # =================================================

            object_bgr = image_bgr[
                by:by + bh,
                bx:bx + bw
            ]

            object_hsv = cv2.cvtColor(
                object_bgr,
                cv2.COLOR_BGR2HSV
            )


            # Tạo mask đúng theo hình contour
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


            detected_fruits.append({
                "label": fruit_label,
                "confidence": round(
                    float(fruit_confidence) * 100,
                    2
                ),
                "area": int(area)
            })
            
            cv2.rectangle(
                result_img,
                (bx, by),
                (bx + bw, by + bh),
                (0, 255, 0),
                2
            )

            cv2.putText(
                result_img,
                f"{fruit_label} {fruit_confidence * 100:.1f}%",
                (bx, max(20, by - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            
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

    hist_norm = hist / (
        hist.sum() + 1e-7
    )

    entropy = -float(
        np.sum([
            p * np.log2(p)
            for p in hist_norm
            if p > 0
        ])
    )

    stats = {
        "mean": round(float(np.mean(gray)), 2),
        "std": round(float(np.std(gray)), 2),
        "variance": round(float(np.var(gray)), 2),
        "min": int(np.min(gray)),
        "max": int(np.max(gray)),
        "entropy": round(entropy, 2)
    }

    return (
        result_img,
        mask_clean,
        count,
        int(total_area),
        stats,
        detected_fruits
    )

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """Render trang chủ giao diện web"""

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )



@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    bins: int = Form(256) # <-- MỚI: Nhận tham số bins từ giao diện HTML
):
    """API Nhận ảnh từ giao diện, phân tích và trả về kết quả"""

    try:
        # Đọc dữ liệu ảnh được upload
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
                "error": "Không thể đọc hoặc xử lý ảnh."
            }
            
        img_resized = cv2.resize(
            img,
            (
                400,
                int(
                    400 *
                    img.shape[0] /
                    img.shape[1]
                )
            )
        )
        
        # Gọi hàm xử lý phân loại cũ
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

        # <-- MỚI: Gọi hàm vẽ Color Histogram ở đây -->
        histogram_b64 = generate_color_histogram(img_resized, bins)

        if detected_fruits:
            best_fruit = max(
                detected_fruits,
                key=lambda x: x["confidence"]
            )
            predicted_label = best_fruit["label"]
            confidence = (
                f'{best_fruit["confidence"]}%'
            )
        else:
            predicted_label = "Unknown"
            confidence = "0%"

        return {
            "filename": file.filename,
            "prediction": predicted_label,
            "confidence": confidence,
            "detected_count": count,
            "total_area": total_area,
            "stats": stats,
            "objects": detected_fruits,
            "histogram_b64": histogram_b64 # <-- MỚI: Trả về thêm chuỗi biểu đồ
        }
        
    except Exception as e:
        return {
            "error": str(e)
        }
