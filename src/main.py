from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import cv2
import numpy as np

app = FastAPI()

# Nạp thư mục chứa giao diện HTML
templates = Jinja2Templates(directory="templates")

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
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}