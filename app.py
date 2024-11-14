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
    label_map = {0: "Cataract", 1: "Diabetic Retinopathy", 2: "Glaucoma", 3: "Normal"}
    return label_map[class_index]


def predict(model, image_tensor, device):
    model.eval()
    model.to(device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor).logits
        predicted_class_index = torch.argmax(outputs, dim=1).item()
    return decode_label(predicted_class_index)


def main():
    st.title("Eye Disease Classification")

    model_path = "model/efficientnet_model.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(model_path, device)

    st.write(
        "Upload an image of an eye to get a prediction on whether it shows cataract, diabetic retinopathy, glaucoma, or is normal."
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        image_tensor = process_image(image)
        predicted_class = predict(model, image_tensor, device)

        st.write(f"Predicted Class: {predicted_class}")


if __name__ == "__main__":
    main()
