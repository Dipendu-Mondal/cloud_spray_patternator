
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from PIL import Image

st.set_page_config(page_title="Cloud Spray Patternator", layout="wide")

st.title("Cloud-Based Spray Patternator")

uploaded = st.file_uploader("Upload Spray Image", type=["png","jpg","jpeg"])

def calculate_angle(lines):
    if lines is None:
        return None
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta)
        angles.append(angle)
    return np.mean(angles)

if uploaded:
    image = Image.open(uploaded)
    image_np = np.array(image)

    st.subheader("Original Image")
    st.image(image_np, use_container_width=True)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.medianBlur(gray, 3)

    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    edges = cv2.Canny(thresh, 50, 150)

    lines = cv2.HoughLines(edges,1,np.pi/180,100)

    angle = calculate_angle(lines)

    st.subheader("Detected Spray Region")
    st.image(edges, use_container_width=True)

    if angle:
        st.metric("Estimated Spray Angle", f"{angle:.2f}°")

    h, w = gray.shape
    sample_y = int(h * 0.5)

    profile = gray[sample_y, :]

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(profile)
    ax.set_title("Spray Distribution Profile")
    ax.set_xlabel("Pixel Position")
    ax.set_ylabel("Intensity")

    st.pyplot(fig)

    peaks, _ = find_peaks(profile, prominence=10)

    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(profile)
    ax2.plot(peaks, profile[peaks], "x")
    ax2.set_title("Solid Stream Jet Detection")
    ax2.set_xlabel("Pixel Position")
    ax2.set_ylabel("Intensity")

    st.pyplot(fig2)

    st.metric("Detected Peaks / Possible Jets", len(peaks))

st.markdown("---")
st.markdown("Built using Streamlit + OpenCV + SciPy")
