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


#
def process_hsv_segmentation(image_bgr, lower_hsv, upper_hsv, min_area):
    # Chuyển đổi BGR sang HSV
    hsv_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Lọc tạo mặt nạ (Mask) dựa trên ngưỡng người dùng chọn
    mask = cv2.inRange(hsv_img, np.array(lower_hsv, dtype=np.uint8), np.array(upper_hsv, dtype=np.uint8))

    # Lọc nhiễu bằng Morphology (Đóng / Mở)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    # Tách vật thể (áp mặt nạ vào ảnh gốc)
    result_segmented = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_clean)
    
    # Đếm số lượng vật thể thỏa mãn min_area
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = sum(1 for cnt in contours if cv2.contourArea(cnt) >= min_area)

    return mask_clean, result_segmented, count

def equalize_image_histogram(image_bgr):
    # Chuyển sang YUV để cân bằng trên kênh độ sáng (Y) thay vì cân bằng sai màu trên RGB
    img_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

#

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
    h_min: int = Form(0), h_max: int = Form(179),
    s_min: int = Form(0), s_max: int = Form(255),
    v_min: int = Form(0), v_max: int = Form(255),
    min_area: int = Form(2000),
    stat_k: float = Form(0.0),
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Resize ảnh để xử lý nhanh hơn
        new_width = 400
        new_height = max(1, int(new_width * img.shape[0] / img.shape[1]))
        img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        before_image_b64 = image_to_base64(img_resized, ".jpg")

        # 1. KỸ THUẬT HSV: Tách vật thể
        lower_hsv = [h_min, s_min, v_min]
        upper_hsv = [h_max, s_max, v_max]
        mask_img, segmented_img, object_count = process_hsv_segmentation(img_resized, lower_hsv, upper_hsv, min_area)
        
        mask_b64 = image_to_base64(mask_img, ".png")
        segmented_b64 = image_to_base64(segmented_img, ".jpg")

        # 2. KỸ THUẬT HISTOGRAM: Cân bằng Histogram
        hist_eq_img = equalize_image_histogram(img_resized)
        hist_eq_b64 = image_to_base64(hist_eq_img, ".jpg")
        
        # Vẽ biểu đồ trước và sau
        hist_before_b64 = generate_color_histogram(img_resized, bins)
        hist_after_b64 = generate_color_histogram(hist_eq_img, bins)

        # 3. KỸ THUẬT THỐNG KÊ (Giữ lại tính năng tính Mean/Std của bạn)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        t_val = float(np.clip(mean_val + stat_k * std_val, 0, 255))
        _, binary_stat = cv2.threshold(gray, t_val, 255, cv2.THRESH_BINARY)
        
        # --- BỔ SUNG ĐÁNH GIÁ HSV ---
        total_pixels = mask_img.shape[0] * mask_img.shape[1]
        hsv_ratio = round((np.count_nonzero(mask_img) / total_pixels) * 100, 2)

        # --- BỔ SUNG ĐÁNH GIÁ HISTOGRAM (TƯƠNG PHẢN) ---
        gray_before = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(hist_eq_img, cv2.COLOR_BGR2GRAY)
        contrast_before = round(float(np.std(gray_before)), 2)
        contrast_after = round(float(np.std(gray_after)), 2)

        return {
            "before_image_b64": before_image_b64,
            "mask_b64": mask_b64,
            "segmented_b64": segmented_b64,
            "hist_eq_b64": hist_eq_b64,
            "hist_before_b64": hist_before_b64,
            "hist_after_b64": hist_after_b64,
            "stat_gray_b64": image_to_base64(gray, ".png"),
            "stat_bin_b64": image_to_base64(binary_stat, ".png"),
            
            "hsv_ratio": hsv_ratio,
            "contrast_before": contrast_before,
            "contrast_after": contrast_after,
            "object_count": object_count,
            "stats": {
                "mean": round(mean_val, 2), "std": round(std_val, 2), "threshold": round(t_val, 2)
            }
        }
    except Exception as e:
        return {"error": str(e)}
 