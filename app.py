import streamlit as st
import cv2
import numpy as np
from PIL import Image

def process_image(image, sensitivity, min_dist_scale):
    # 1. 轉成 OpenCV 格式
    img_cv = np.array(image)
    if img_cv.shape[2] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)
    
    # 2. 轉灰階並增強對比 (CLAHE) - 讓藥丸跟背景分更開
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # 3. 二值化 (閾值處理)
    # 這裡用 slider 的數值來決定「多亮的東西才算藥丸」
    _, thresh = cv2.threshold(gray, sensitivity, 255, cv2.THRESH_BINARY_INV)

    # 4. 去除雜訊 (開運算) - 把小的反光點吃掉
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 5. 確保背景是乾淨的 (膨脹)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 6. 找中心點 (距離變換) - 這是分開沾黏藥丸的關鍵
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # 7. 根據中心點的「高峰」來決定哪裡是藥丸的核心
    # min_dist_scale 越小，越容易把黏在一起的算成多顆
    _, sure_fg = cv2.threshold(dist_transform, min_dist_scale * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 8. 算數量
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = len(contours)
    
    # 畫圖 (只畫中心點，因為邊緣已經不準了)
    output_img = img_cv.copy()
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        # 畫個紅點在中心
        cv2.circle(output_img, (int(x+w/2), int(y+h/2)), 5, (0, 0, 255), -1)
        # 畫個綠框
        cv2.rectangle(output_img, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 0), 2)

    return output_img, count, opening, dist_transform

# --- 介面 ---
st.title("💊 藥丸計數器 V2 (抗反光版)")
st.info("💡 小撇步：如果藥丸在袋子裡，請盡量**拉平袋子**減少皺褶反光。")

uploaded_file = st.file_uploader("上傳照片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='原始圖片', use_column_width=True)
    
    st.write("---")
    st.subheader("🎛️ 參數調整 (調到準為止)")
    
    col1, col2 = st.columns(2)
    with col1:
        # 控制二值化：數值越小，只有越深色的東西會被抓到（適合淺色藥丸深色背景）
        # 如果是黃藥丸(淺)在深桌子上，通常要反過來，或調整這個值
        thresh_val = st.slider("1. 顏色過濾閾值 (過濾背景)", 0, 255, 120)
    with col2:
        # 控制沾黏分離程度
        dist_scale = st.slider("2. 分離沾黏強度 (越小分越細)", 0.1, 0.9, 0.5)

    if st.button('開始計算'):
        result_img, count, debug_mask, debug_dist = process_image(image, thresh_val, dist_scale)
        
        st.success(f"📊 估計數量： {count} 顆")
        st.image(result_img, caption='計算結果 (紅點為判定核心)', use_column_width=True)
        
        with st.expander("👀 查看電腦看到了什麼 (除錯用)"):
            st.write("這是電腦過濾後的黑白影像，藥丸應該要是白色的，背景是黑色的。如果這張圖很亂，請調整上面的「顏色過濾閾值」。")
            st.image(debug_mask, caption='黑白遮罩', use_column_width=True, clamp=True)
