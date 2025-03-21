import streamlit as st
import numpy as np
import joblib
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt  # For debugging

# Load the trained model
try:
    model = joblib.load("Logistik.pkl")
except FileNotFoundError:
    st.error("Modellen kunde inte hittas! Kontrollera att filen 'random_forest_mnist.pkl' finns i mappen.")
    st.stop()

# Streamlit UI
st.title("Draw a digit and app gissar det!")

# Create canvas for drawing
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Prediction logic
if st.button("guess the digit") and model is not None:
    if canvas_result.image_data is not None:
        # Convert to NumPy array
        img_array = np.array(canvas_result.image_data)

        # Check if the image contains any drawing
        if np.sum(img_array) == 0:
            st.warning("Ingen siffra ritad! Försök igen.")
        else:
            # Convert to grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)

          
            # Ensure brightness normalization
            img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)

            # Resize to 28x28 pixels (same as MNIST dataset)
            img_array = cv2.resize(img_array, (28, 28))

            # Normalize pixel values (0 to 1 range)
            img_array = img_array / 255.0

            # Flatten and reshape for model input
            img_array = img_array.flatten().reshape(1, -1)

            # Predict using the trained model
            prediction = model.predict(img_array)

            # Display the prediction
            st.write(f"Modellen gissar att detta är en: **{prediction[0]}**")

            # Debugging: Show the processed image
            fig, ax = plt.subplots()
            ax.imshow(img_array.reshape(28, 28), cmap="gray")
            ax.set_title("Processed Image Sent to Model")
            st.pyplot(fig)

    else:
        st.warning("Rita en siffra innan du klickar på 'Prediktera'!")
