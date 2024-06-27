import cv2
import numpy as np
from PIL import Image
import streamlit as st
MODEL = "model/MobileNetSSD_deploy.caffemodel"
PROTOTXT = "model/MobileNetSSD_deploy.prototxt.txt"


def process_image(image):
    # Tạo blob từ hình ảnh
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
    )
    # Đọc mô hình từ file
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    # Đặt blob làm đầu vào cho mạng
    net.setInput(blob)
    # Thực hiện dự đoán
    detections = net.forward()
    return detections


def annotate_image(image, detections, confidence_threshold=0.5):
    # Lấy kích thước của hình ảnh
    (h, w) = image.shape[:2]
    # Lặp qua các phát hiện
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # Kiểm tra ngưỡng độ tin cậy
        if confidence > confidence_threshold:
            # Lấy chỉ số của nhãn lớp từ phát hiện
            idx = int(detections[0, 0, i, 1])
            # Tính toán tọa độ của hộp giới hạn
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # Vẽ hình chữ nhật xung quanh đối tượng
            cv2.rectangle(image, (startX, startY),
                          (endX, endY), (70, 70, 70), 2)
    return image


def detect_objects(image_path):
    # Đọc hình ảnh từ file
    image = cv2.imread(image_path)
    # Xử lý hình ảnh
    detections = process_image(image)
    # Vẽ hình chữ nhật xung quanh đối tượng
    annotated_image = annotate_image(image.copy(), detections)
    # Hiển thị hình ảnh
    cv2.imshow("Output", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    st.title('Object Detection for Images')
    file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])

    if file is not None:
        st.image(file, caption="Uploaded Image")

        try:
            image = Image.open(file)
            image = np.array(image)

            if image.size == 0:
                st.error("Uploaded image is empty.")
                return

            detections = process_image(image)
            processed_image = annotate_image(image, detections)

            st.image(processed_image, caption="Processed Image")
        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
