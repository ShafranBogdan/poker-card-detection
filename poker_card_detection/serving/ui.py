import requests
import streamlit as st

API_URL = "http://localhost:8090/detect"

st.title("Poker Card Detection")
st.write("Upload a photo of playing cards to detect and classify the poker hand.")

mode = st.radio("Input mode", ["Upload image", "Camera"])

image_bytes = None

if mode == "Upload image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        st.image(image_bytes, caption="Uploaded image", use_column_width=True)

elif mode == "Camera":
    camera_photo = st.camera_input("Take a photo")
    if camera_photo is not None:
        image_bytes = camera_photo.getvalue()

if image_bytes is not None and st.button("Detect"):
    with st.spinner("Running detection..."):
        response = requests.post(
            API_URL, files={"file": ("image.jpg", image_bytes, "image/jpeg")}
        )

    if response.status_code == 200:
        result = response.json()
        st.success(f"Poker Hand: **{result['poker_hand']}**")
        st.metric("Cards detected", result["cards_found"])
        st.json(result["detections"])
    else:
        st.error(f"Error {response.status_code}: {response.text}")


if __name__ == "__main__":
    pass
