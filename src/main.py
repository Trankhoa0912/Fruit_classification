from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import cv2
import numpy as np

app = FastAPI()

# Nạp thư mục chứa giao diện HTML
templates = Jinja2Templates(directory="templates")

def process_statistical(image_bgr, sample_bbox=None, k_factor=1.8, min_area=2000):
    h_s, w_s = image_bgr.shape[:2]

    x, y, w, h = sample_bbox if sample_bbox else (int(w_s*0.25), int(h_s*0.25), int(w_s*0.5), int(h_s*0.5))
    roi_hsv = cv2.cvtColor(image_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
    valid_pixels = roi_hsv[(roi_hsv[:, :, 1] > 40) & (roi_hsv[:, :, 2] > 40)]
    
    mean_hsv = np.mean(valid_pixels, axis=0) if len(valid_pixels) >= 30 else np.mean(roi_hsv.reshape(-1, 3), axis=0)
    std_hsv = np.maximum(np.std(valid_pixels, axis=0) if len(valid_pixels) >= 30 else np.std(roi_hsv.reshape(-1, 3), axis=0), [5.0, 20.0, 20.0])

  
    hsv_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.clip(mean_hsv - k_factor * std_hsv, [0, 50, 50], [179, 255, 255]).astype(np.uint8)
    upper = np.clip(mean_hsv + k_factor * std_hsv, [0, 50, 50], [179, 255, 255]).astype(np.uint8)
    mask = cv2.inRange(hsv_img, lower, upper)

   
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_clean = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_img = image_bgr.copy()
    count, total_area = 0, 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            count += 1
            total_area += area
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cv2.rectangle(result_img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.putText(result_img, f"Obj {count}", (bx, max(20, by - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist / (hist.sum() + 1e-7)
    entropy = -float(np.sum([p * np.log2(p) for p in hist_norm if p > 0]))

    stats = {
        "mean": round(float(np.mean(gray)), 2),
        "std": round(float(np.std(gray)), 2),
        "variance": round(float(np.var(gray)), 2),
        "min": int(np.min(gray)),
        "max": int(np.max(gray)),
        "entropy": round(entropy, 2)
    }

    return result_img, mask_clean, count, int(total_area), stats

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """Render trang chủ giao diện web"""
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """API Nhận ảnh từ giao diện, phân tích và trả về kết quả"""
    try:
        # Đọc dữ liệu ảnh được upload
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Không thể đọc hoặc xử lý ảnh."}

        # ---------------------------------------------------------
        # CHÈN CODE XỬ LÝ ẢNH & MODEL CỦA NHÓM VÀO ĐÂY
        # Dựa trên code jupyter của bạn, thay đổi kích thước ảnh:
        img_resized = cv2.resize(img, (400, int(400 * img.shape[0] / img.shape[1])))

        result_img, mask_clean, count, total_area, stats = process_statistical(
                    image_bgr=img_resized,
                    sample_bbox=None,
                    k_factor=1.8,
                    min_area=2000
                )
        
        # Ví dụ: 
        # 1. Trích xuất đặc trưng Color Histogram từ 'img_resized'
        # 2. Đưa vào mô hình dự đoán: label = model.predict(features)
        # ---------------------------------------------------------

        # Đây là kết quả mô phỏng (dummy) trả về giao diện.
        # Khi tích hợp model thật, hãy đổi 'Apple' thành biến 'label' của bạn.
        predicted_label = "Apple" 
        confidence = "92.5%"

        return {
            "filename": file.filename,
            "prediction": predicted_label,
            "confidence": confidence,
            "detected_count": count,
            "total_area": total_area,
            "stats": stats
        }

    except Exception as e:
        return {"error": str(e)}