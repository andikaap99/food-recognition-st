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

    with st.expander('Macronutrient Detail'):
        match predicted_class_index:
            case 0:
                st.write('Per 100 gr ')
                st.write('Calori    : 250 kcal')
                st.write('Protein   : 25 gr')
                st.write('Fat       : 15 gr')
            case 1:
                st.write('Per 120 gr ')
                st.write('Calori    : 295 kcal')
                st.write('Protein   : 17 gr')
                st.write('Fat       : 14 gr')
            case 2:
                st.write('Per 100 gr ')
                st.write('Calori    : 312 kcal')
                st.write('Protein   : 3.4 gr')
                st.write('Fat       : 15 gr')
            case 3:
                st.write('Per 200 gr ')
                st.write('Calori    : 345 kcal')
                st.write('Protein   : 15 gr')
                st.write('Fat       : 18 gr')
            case 4:
                st.write('Per 100 gr ')
                st.write('Calori    : 260 kcal')
                st.write('Protein   : 28 gr')
                st.write('Fat       : 12 gr')
            case 5:
                st.write('Per 200 gr ')
                st.write('Calori    : 410 kcal')
                st.write('Protein   : 10 gr')
                st.write('Fat       : 17 gr')
            case 6:
                st.write('Per 200 gr ')
                st.write('Calori    : 400 kcal')
                st.write('Protein   : 9 gr')
                st.write('Fat       : 18 gr')
            case 7:
                st.write('Per 300 gr ')
                st.write('Calori    : 700 kcal')
                st.write('Protein   : 20 gr')
                st.write('Fat       : 35 gr')
            case 8:
                st.write('Per 100 gr ')
                st.write('Calori    : 285 kcal')
                st.write('Protein   : 12 gr')
                st.write('Fat       : 10 gr')
            case 9:
                st.write('Per 250 gr ')
                st.write('Calori    : 320 kcal')
                st.write('Protein   : 25 gr')
                st.write('Fat       : 18 gr')
            case 10:
                st.write('Per 150 gr ')
                st.write('Calori    : 470 kcal')
                st.write('Protein   : 28 gr')
                st.write('Fat       : 30 gr')
            case 11:
                st.write('Per 10 skewers ')
                st.write('Calori    : 350 kcal')
                st.write('Protein   : 26 gr')
                st.write('Fat       : 22 gr')
            case 12:
                st.write('Per 250 gr ')
                st.write('Calori    : 275 kcal')
                st.write('Protein   : 18 gr')
                st.write('Fat       : 10 gr')