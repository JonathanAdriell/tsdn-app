import streamlit as st
import torch
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from transformers import AutoModelForImageClassification
from torchvision import transforms
from PIL import Image

LABELS = ["Katarak", "Retinopati Diabetik", "Glaukoma", "Normal"]


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


def predict_with_probabilities(model, image_tensor, device):
    model.eval()
    model.to(device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor).logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
    predicted_class_index = torch.argmax(outputs, dim=1).item()
    return probabilities, LABELS[predicted_class_index]


def plot_probabilities(probabilities):
    sorted_indices = np.argsort(probabilities)
    sorted_probs = np.array(probabilities)[sorted_indices]
    sorted_labels = np.array(LABELS)[sorted_indices]

    normalized_probs = (sorted_probs - np.min(sorted_probs)) / (
        np.max(sorted_probs) - np.min(sorted_probs)
    )
    colors = cm.viridis(normalized_probs)

    plt.figure(figsize=(8, 5))
    plt.barh(sorted_labels, sorted_probs, color=colors)
    plt.xlabel("Probabilitas")
    plt.ylabel("Kondisi Mata")
    plt.title("Probabilitas Prediksi untuk Setiap Kondisi Mata")

    legend_elements = [
        Patch(
            facecolor=colors[i],
            edgecolor="none",
            label=f"{sorted_labels[i]}: {sorted_probs[i]:.2f}",
        )
        for i in range(len(sorted_labels))
    ]

    plt.legend(
        handles=legend_elements[::-1],
        loc="upper right",
        bbox_to_anchor=(1.01, -0.2),
        title="Probabilitas",
    )
    plt.tight_layout()

    st.pyplot(plt)


def generate_saliency_map(model, image_tensor, predicted_index, device):
    model.eval()
    model.to(device)
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad_()

    # Forward pass
    outputs = model(image_tensor)
    class_score = outputs.logits[0, predicted_index]

    # Backward pass to get the gradients
    model.zero_grad()
    class_score.backward()

    gradients = image_tensor.grad[0].cpu().detach().numpy()
    saliency_map = np.max(np.abs(gradients), axis=0)

    return saliency_map


def plot_saliency_map(saliency_map):
    plt.figure(figsize=(8, 5))
    plt.imshow(saliency_map, cmap="turbo")
    plt.axis("off")
    plt.title("Saliency Map")
    st.pyplot(plt)


def interpret_saliency_map(predicted_index):
    saliency_explanations = {
        0: (
            "Pada gambar fundus dengan katarak, saliency map menyoroti area buram pada mata, "
            "yang ditunjukkan dengan warna yang lebih terang. Hal ini menunjukkan bahwa model memperhatikan "
            "wilayah di mana penurunan ketajaman penglihatan terjadi akibat katarak. Secara visual, mata yang "
            "terkena katarak cenderung terlihat buram dibandingkan dengan fundus mata pada kondisi lain."
        ),
        1: (
            "Pada gambar fundus dengan retinopati diabetik, saliency map menyoroti pembuluh darah yang lebih jelas "
            "atau menonjol. Hal ini menunjukkan bahwa model memanfaatkan informasi pembuluh darah sebagai indikator awal "
            "retinopati diabetik, yang sesuai dengan kenyataan karena retinopati diabetik seringkali menyebabkan perubahan "
            "pada pembuluh darah retina, seperti penyumbatan atau pembentukan pembuluh darah baru. Pembuluh darah baru ini cenderung lebih rapuh "
            "dan rentan pecah jika tidak berkembang dengan baik, yang dapat menyebabkan perdarahan retina."
        ),
        2: (
            "Pada gambar fundus dengan glaukoma, saliency map menyoroti area sekitar cakram optik (diskus optikus). "
            "Hal ini sesuai kenyataan bahwa glaukoma biasanya berkaitan dengan tekanan intraokular yang dapat "
            "memengaruhi bentuk dan kesehatan cakram optik. Hal ini mengindikasikan bahwa model menggunakan struktur "
            "cakram optik yang membesar sebagai indikator awal glaukoma."
        ),
        3: (
            "Pada gambar fundus mata normal, saliency map tidak terfokus pada area tertentu, melainkan menunjukkan "
            "penyebaran yang lebih merata di seluruh retina. Model mempelajari bahwa mata dalam kondisi normal tidak menunjukkan "
            "kelainan atau perubahan mencolok pada struktur retina, yang menyebabkan titik terang pada saliency map tersebar secara merata "
            "tanpa fokus pada satu area tertentu."
        ),
    }
    return saliency_explanations[predicted_index]


def main():
    st.set_page_config(
        page_title="PANDAWA - Eye Disease Detection",
        page_icon="👁️",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        "<h1 style='text-align: center; color: #007ACC;'>PANDAWA</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h2 style='text-align: center; color: #007ACC;'>Pendeteksi Awal Penyakit Mata</h2>",
        unsafe_allow_html=True,
    )
    st.subheader("Penerapan Model ML EfficientNet untuk Deteksi Dini Penyakit Mata")

    # Introduction
    st.write(
        """
        Selamat datang di Aplikasi Klasifikasi Penyakit Mata! Kami hadir untuk membantu deteksi dini kondisi kesehatan mata menggunakan teknologi machine learning.
        Aplikasi ini mampu mengklasifikasikan gambar retina dalam empat kategori:
        """
    )
    st.markdown(
        """
        - **Katarak**
        - **Retinopati Diabetik**
        - **Glaukoma**
        - **Normal**
        """,
        unsafe_allow_html=True,
    )

    st.write(
        "Deteksi dini ini penting untuk mencegah kebutaan dan meningkatkan kualitas hidup pasien."
    )

    # Tabs for more information about diseases
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Latar Belakang",
            "Mengenai Katarak",
            "Mengenai Retinopati Diabetik",
            "Mengenai Glaukoma",
        ]
    )

    with tab1:
        st.markdown(
            """
            > “Early detection and treatment of eye diseases such as cataracts, diabetic retinopathy, and glaucoma are crucial in preventing vision loss. Regular eye examinations can identify these conditions before significant damage occurs, allowing for timely intervention and better outcomes.”
            > 
            > — Dr. Paul Sieving, former Director of the National Eye Institute (2019)
            """,
            unsafe_allow_html=True,
        )

        st.write("## Penyakit Mata di Indonesia")
        st.write(
            """
            Di Indonesia, kebutaan menjadi masalah kesehatan yang signifikan. Penyebab utama kebutaan meliputi **katarak**, **glaukoma**, dan **retinopati diabetik**. Namun, sekitar 95% kasus kebutaan dapat dicegah melalui deteksi dini dan pengobatan yang tepat (Dinkes DIY, 2024).
            - **Glaukoma**: Prevalensi glaukoma di Indonesia diperkirakan mencapai 0,46% (Kementerian Kesehatan, 2019). Secara global, jumlah penderita glaukoma pada 2020 mencapai 76 juta orang.
            - **Retinopati Diabetik**: Sekitar 42,6% penderita diabetes di Indonesia mengalami retinopati diabetik, dan 10% di antaranya berujung pada kebutaan.
            - **Katarak**: Menyumbang 75% dari kasus kebutaan, dengan lebih dari 1,2 juta kasus baru setiap tahunnya (WHO, 2022).
            """
        )

        st.write("## Potensi Machine Learning untuk Deteksi Dini")
        st.write(
            """
            Machine learning menawarkan potensi besar dalam mendeteksi penyakit mata sejak dini dengan menganalisis gambar retina secara otomatis. Dengan kemampuan untuk mengenali pola-pola halus yang menunjukkan perubahan pada retina, machine learning dapat membantu mengidentifikasi tanda-tanda awal penyakit mata seperti **katarak**, **glaukoma**, dan **retinopati diabetik**, bahkan sebelum gejala klinis muncul. 
            
            Teknologi ini memungkinkan deteksi lebih cepat dan lebih akurat, yang dapat mempercepat penanganan medis dan mengurangi risiko kebutaan. Dengan penggunaan machine learning, pemeriksaan mata dapat dilakukan secara lebih efisien, mengurangi beban pada tenaga medis dan meningkatkan akses bagi lebih banyak pasien.
            """
        )

    with tab2:
        st.image(
            Image.open("assets/katarak.png"),
            caption="Gambar Retina untuk Katarak",
            use_container_width=True,
        )
        st.write(
            """
            **Katarak** adalah kondisi di mana lensa mata menjadi keruh, menghalangi cahaya mencapai retina dan menyebabkan penglihatan kabur. Deteksi katarak dapat dilakukan melalui **analisis gambar retina**, dengan mencari tanda-tanda kekeruhan pada lensa mata. **Machine learning** dapat menganalisis gambar ini dan mendeteksi katarak pada tahap awal, memungkinkan pengobatan yang lebih efektif.
            """
        )

    with tab3:
        left_col, center_col, right_col = st.columns([1, 4, 1])
        with center_col:
            st.image(
                Image.open("assets/retinopati_diabetik.jpeg"),
                caption="Gambar Retina untuk Retinopati Diabetik",
                width=450,
            )

        st.write(
            """
            **Retinopati diabetik** adalah komplikasi mata pada penderita diabetes yang terjadi akibat kerusakan pembuluh darah di retina. Gejala yang dapat dikenali meliputi **pembuluh darah yang bocor** atau **perubahan bentuk pembuluh darah**. Deteksi dini retinopati diabetik sangat penting untuk mencegah kebutaan, dan machine learning dapat membantu dalam proses deteksi ini melalui gambar retina.
            """
        )

    with tab4:
        st.image(
            Image.open("assets/glaukoma.png"),
            caption="Gambar Retina untuk Glaukoma",
            use_container_width=True,
        )
        st.write(
            """
            **Glaukoma** adalah penyakit mata yang terjadi ketika tekanan dalam mata meningkat, merusak saraf optik dan berisiko menyebabkan kebutaan. Deteksi glaukoma dapat dilakukan dengan mengukur perubahan dalam **rasio Optic Cup (OC)** dan **Optic Disc (OD)**. Machine learning dapat menganalisis gambar retina untuk mendeteksi perubahan kecil ini, yang merupakan tanda awal glaukoma.
            """
        )

    st.markdown("<hr style='border: 1px solid #007ACC;'>", unsafe_allow_html=True)
    st.write(
        """
        Cukup unggah gambar retina Anda, dan sistem kami akan menganalisisnya untuk mengidentifikasi salah satu dari empat kondisi mata: **katarak**, **retinopati diabetik**, **glaukoma**, atau **normal**.
        """
    )

    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang Diunggah", use_container_width=True)

        model_path = "model/efficientnet_model.pth"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(model_path, device)

        image_tensor = process_image(image)
        probabilities, predicted_class = predict_with_probabilities(
            model, image_tensor, device
        )

        st.write(f"**Prediksi Kondisi Mata**: {predicted_class}")
        st.write("Dengan probabilitas untuk setiap kondisi mata:")
        plot_probabilities(probabilities)

        st.write(
            "Saliency map menunjukkan bagian-bagian gambar yang paling mempengaruhi keputusan model dalam membuat prediksi. Berikut adalah saliency map berdasarkan prediksi model:"
        )
        predicted_index = LABELS.index(predicted_class)
        saliency_map = generate_saliency_map(
            model, image_tensor, predicted_index, device
        )
        plot_saliency_map(saliency_map)

        saliency_description = interpret_saliency_map(predicted_index)
        st.write(saliency_description)
        st.write(
            "Dengan saliency map ini, dokter juga dapat memverifikasi hasil model, memastikan bahwa area yang terdeteksi benar-benar relevan secara medis dan mendukung pengambilan keputusan yang lebih tepat."
        )


if __name__ == "__main__":
    main()
