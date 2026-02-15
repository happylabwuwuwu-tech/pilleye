import streamlit as st
import cv2
import numpy as np
from PIL import Image

def count_pills(image, min_area=100):
    img_cv = np.array(image)
    if img_cv.shape[2] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edged = cv2.Canny(blurred, 30, 150)
    dilated = cv2.dilate(edged, None, iterations=2)
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    output_img = img_cv.copy()
    
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        count += 1
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output_img, str(count), (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return output_img, count

st.title("💊 簡易藥丸計數器")
st.write("請上傳藥丸照片（注意：不要重疊，背景單純）")
sensitivity = st.slider("靈敏度調整", 10, 2000, 300)
uploaded_file = st.file_uploader("上傳圖片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='原始圖片', use_column_width=True)
    if st.button('開始計數'):
        result_img, total_count = count_pills(image, sensitivity)
        st.success(f"共偵測到 {total_count} 顆")
        st.image(result_img, caption='結果', use_column_width=True)
