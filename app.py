import streamlit as st
import torch
from transformers import AutoModelForImageClassification
from torchvision import transforms
from PIL import Image


def load_model(model_path, device):
    model = AutoModelForImageClassification.from_pretrained("google/efficientnet-b7")
    model.classifier = torch.nn.Linear(model.classifier.in_features, 4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def process_image(image):
    image = image.convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


def decode_label(class_index):
    label_map = {0: "Karatak", 1: "Retinopati Diabetik", 2: "Glaukoma", 3: "Normal"}
    return label_map.get(class_index, "Unknown")


def predict(model, image_tensor, device):
    model.eval()
    model.to(device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor).logits
        predicted_class_index = torch.argmax(outputs, dim=1).item()
    return decode_label(predicted_class_index)


def main():
    st.set_page_config(
        page_title="PANDAWA - Eye Disease Detection",
        page_icon="👁️",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    st.markdown("<h1 style='text-align: center; color: #007ACC;'>PANDAWA</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #007ACC;'>Pendeteksi Awal Penyakit Mata</h2>", unsafe_allow_html=True)
    st.subheader("Penerapan Model ML EfficientNet untuk Deteksi Dini Penyakit Mata")

    # Introduction
    st.write("""
    Selamat datang di Aplikasi Klasifikasi Penyakit Mata! Kami hadir untuk membantu deteksi dini kondisi kesehatan mata menggunakan teknologi machine learning.
    Aplikasi ini mampu mengklasifikasikan gambar retina dalam empat kategori:
    """)
    st.markdown("""
    - **Katarak**
    - **Retinopati Diabetik**
    - **Glaukoma**
    - **Normal**
    """, unsafe_allow_html=True)

    st.write("Deteksi dini ini penting untuk mencegah kebutaan dan meningkatkan kualitas hidup pasien.")

    # Tabs for more information about diseases
    tab1, tab2, tab3, tab4 = st.tabs(["Latar Belakang", "Mengenai Katarak", "Mengenai Retinopati Diabetik", "Mengenai Glaukoma"])

    with tab1:
        st.markdown("""
        > “Early detection and treatment of eye diseases such as cataracts, diabetic retinopathy, and glaucoma are crucial in preventing vision loss. Regular eye examinations can identify these conditions before significant damage occurs, allowing for timely intervention and better outcomes.”
        > 
        > — Dr. Paul Sieving, former Director of the National Eye Institute (2019)
        """, unsafe_allow_html=True)

        st.write("## Penyakit Mata di Indonesia")
        st.write("""
        Di Indonesia, kebutaan menjadi masalah kesehatan yang signifikan. Penyebab utama kebutaan meliputi **katarak**, **glaukoma**, dan **retinopati diabetik**. Namun, sekitar 95% kasus kebutaan dapat dicegah melalui deteksi dini dan pengobatan yang tepat (Dinkes DIY, 2024).
        - **Glaukoma**: Prevalensi glaukoma di Indonesia diperkirakan mencapai 0,46% (Kementerian Kesehatan, 2019). Secara global, jumlah penderita glaukoma pada 2020 mencapai 76 juta orang.
        - **Retinopati Diabetik**: Sekitar 42,6% penderita diabetes di Indonesia mengalami retinopati diabetik, dan 10% di antaranya berujung pada kebutaan.
        - **Katarak**: Menyumbang 75% dari kasus kebutaan, dengan lebih dari 1,2 juta kasus baru setiap tahunnya (WHO, 2022).
        """)
        
        st.write("## Potensi Machine Learning untuk Deteksi Dini")
        st.write("""
        Machine learning menawarkan potensi besar dalam mendeteksi penyakit mata sejak dini dengan menganalisis gambar retina secara otomatis. Dengan kemampuan untuk mengenali pola-pola halus yang menunjukkan perubahan pada retina, machine learning dapat membantu mengidentifikasi tanda-tanda awal penyakit mata seperti **katarak**, **glaukoma**, dan **retinopati diabetik**, bahkan sebelum gejala klinis muncul. 
        
        Teknologi ini memungkinkan deteksi lebih cepat dan lebih akurat, yang dapat mempercepat penanganan medis dan mengurangi risiko kebutaan. Dengan penggunaan machine learning, pemeriksaan mata dapat dilakukan secara lebih efisien, mengurangi beban pada tenaga medis dan meningkatkan akses bagi lebih banyak pasien.
        """)

    with tab2:
        st.write("## Bagaimana Katarak Dideteksi")
        st.write("""
        **Katarak** adalah kondisi di mana lensa mata menjadi keruh, menghalangi cahaya mencapai retina dan menyebabkan penglihatan kabur. Deteksi katarak dapat dilakukan melalui **analisis gambar retina**, dengan mencari tanda-tanda kekeruhan pada lensa mata. **Machine learning** dapat menganalisis gambar ini dan mendeteksi katarak pada tahap awal, memungkinkan pengobatan yang lebih efektif.
        """)

    with tab3:
        st.write("## Bagaimana Glaukoma Dideteksi")
        st.write("""
        **Glaukoma** adalah penyakit mata yang terjadi ketika tekanan dalam mata meningkat, merusak saraf optik dan berisiko menyebabkan kebutaan. Deteksi glaukoma dapat dilakukan dengan mengukur perubahan dalam **rasio Optic Cup (OC)** dan **Optic Disc (OD)**. Machine learning dapat menganalisis gambar retina untuk mendeteksi perubahan kecil ini, yang merupakan tanda awal glaukoma.
        """)

    with tab4:
        st.write("## Bagaimana Retinopati Diabetik Dideteksi")
        st.write("""
        **Retinopati diabetik** adalah komplikasi mata pada penderita diabetes yang terjadi akibat kerusakan pembuluh darah di retina. Gejala yang dapat dikenali meliputi **pembuluh darah yang bocor** atau **perubahan bentuk pembuluh darah**. Deteksi dini retinopati diabetik sangat penting untuk mencegah kebutaan, dan machine learning dapat membantu dalam proses deteksi ini melalui gambar retina.
        """)

    st.markdown("<hr style='border: 1px solid #007ACC;'>", unsafe_allow_html=True)
    st.write("""
    Cukup unggah gambar retina Anda, dan sistem kami akan menganalisisnya untuk mengidentifikasi salah satu dari empat kondisi mata: **katarak**, **retinopati diabetik**, **glaukoma**, atau **normal**.
    """)

    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang Diunggah", use_container_width=True)

        model_path = "model/efficientnet_model.pth"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(model_path, device)

        image_tensor = process_image(image)
        predicted_class = predict(model, image_tensor, device)

        st.write(f"**Prediksi Kelas**: {predicted_class}")

if __name__ == "__main__":
    main()
