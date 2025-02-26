import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


# Load your model
model = load_model('model_2.keras')

# Fungsi untuk melakukan prediksi
def predict(image):
    image = image.resize((224, 224))  # Sesuaikan dengan ukuran input model
    img_array = np.array(image) / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimension
    prediction = model.predict(img_array)
    return prediction

# UI dengan Streamlit
st.title("Food Recognition App 🍔🍕🍣")

# Tambahkan opsi untuk memilih sumber gambar
option = st.radio("Pilih sumber gambar:", ('Upload Gambar', 'Gunakan Kamera'))

if option == 'Upload Gambar':
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)

elif option == 'Gunakan Kamera':
    captured_image = st.camera_input("Ambil gambar dari kamera")
    if captured_image is not None:
        img = Image.open(captured_image)

if 'img' in locals():
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize image agar sesuai dengan input model
    img = img.resize((224, 224))

    imgplot = plt.imshow(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    labels = ['Ayam Goreng', 'Burger', 'French Fries', 'Gado-Gado', 'Ikan Goreng', 'Mie Goreng', 'Nasi Goreng', 'Nasi Padang', 'Pizza', 'Rawon', 'Rendang', 'Sate', 'Soto']
    classes = model.predict(images)

    predicted_class_index = np.argmax(classes[0])
    predicted_label = labels[predicted_class_index]

    st.write("Prediction:", predicted_label)
    st.write("Classes:", classes)

