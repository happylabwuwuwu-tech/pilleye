import streamlit as st
import cv2
import numpy as np
from PIL import Image

def detect_circles_metal(image, min_dist, param1, param2, min_radius, max_radius):
    # 1. 轉成 OpenCV 格式
    img_cv = np.array(image)
    output_img = img_cv.copy()
    
    if img_cv.shape[2] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)
    
    # 2. 轉灰階
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
    # [金屬盤專用] 3. CLAHE 對比度增強 
    # 這能讓白色藥丸從銀色盤子上「跳」出來
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray)
    
    # 4. 模糊化 (去除金屬髮絲紋與刮痕)
    gray_blurred = cv2.GaussianBlur(gray_enhanced, (9, 9), 2)
    
    # 5. 霍夫圓變換
    # param1 (新增): 邊緣偵測閾值。設得越高，只有非常明顯的邊緣才算（過濾反光）。
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist,
                               param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)
    
    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        count = len(circles[0, :])
        
        for i in circles[0, :]:
            # 畫外圓 (綠色)
            cv2.circle(output_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # 畫圓心 (紅色)
            cv2.circle(output_img, (i[0], i[1]), 2, (0, 0, 255), 3)

    return output_img, count, gray_blurred

# --- 介面 ---
st.title("💊 藥丸計數器 V3.1 (金屬盤專用)")
st.info("💡 針對**反光金屬盤**優化。請調整下方參數直到綠色圈圈剛好套住藥丸。")

uploaded_file = st.file_uploader("上傳照片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='原始圖片', use_column_width=True)
    
    st.write("---")
    st.subheader("🎛️ 金屬盤參數調校")
    
    col1, col2 = st.columns(2)
    with col1:
        # 針對金屬反光，這個要調高！
        param1 = st.slider("1. 邊緣銳利度 (抗反光)", 50, 200, 100, help="數值越高，越不容易被金屬反光騙，但太高會抓不到藥丸邊緣")
        # 圓形判定
        param2 = st.slider("2. 圓形嚴格度", 10, 100, 30, help="數值越小越敏感，數值越大越嚴格")
        
    col3, col4 = st.columns(2)
    with col3:
        min_dist = st.slider("3. 最小間距", 10, 100, 25)
    with col4:
        min_radius = st.slider("4. 最小藥丸半徑", 0, 50, 10)
        max_radius = st.slider("5. 最大藥丸半徑", 20, 150, 60)

    if st.button('開始計算'):
        result_img, count, debug_gray = detect_circles_metal(image, min_dist, param1, param2, min_radius, max_radius)
        
        st.success(f"📊 估計數量： {count} 顆")
        st.image(result_img, caption='計算結果', use_column_width=True)
        
        with st.expander("👀 電腦看到的處理後影像 (檢查對比度)"):
            st.image(debug_gray, caption='增強對比後的灰階圖', use_column_width=True, clamp=True)
