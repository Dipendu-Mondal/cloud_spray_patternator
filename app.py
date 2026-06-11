
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from PIL import Image

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Cloud Spray Patternator",
    layout="wide"
)

st.title("Cloud-Based Spray Patternator")

uploaded = st.file_uploader(
    "Upload Spray Image",
    type=["png", "jpg", "jpeg"]
)

# =====================================================
# MAIN PROCESSING
# =====================================================

if uploaded:

    # =================================================
    # LOAD IMAGE
    # =================================================

    image = Image.open(uploaded).convert("RGB")

    image_np = np.array(image)

    st.subheader("Original Image")

    st.image(
        image_np,
        use_container_width=True
    )

    # =================================================
    # PREPROCESSING
    # =================================================

    gray = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Adaptive thresholding

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # Edge detection

    edges = cv2.Canny(
        thresh,
        50,
        150
    )

    # =================================================
    # REMOVE NOZZLE REGION
    # =================================================

    h, w = edges.shape

    crop_start = int(h * 0.20)

    cropped_edges = edges[crop_start:, :]

    # =================================================
    # LINE DETECTION
    # =================================================

    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=100,
        maxLineGap=20
    )

    line_image = np.zeros_like(cropped_edges)

    left_angles = []

    right_angles = []

    # =================================================
    # DETECT SPRAY BOUNDARY LINES
    # =================================================

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line[0]

            # Calculate line angle

            angle = np.degrees(
                np.arctan2(
                    (y2 - y1),
                    (x2 - x1)
                )
            )

            # LEFT boundary filter

            if -80 < angle < -10:

                left_angles.append(angle)

                cv2.line(
                    line_image,
                    (x1, y1),
                    (x2, y2),
                    255,
                    2
                )

            # RIGHT boundary filter

            elif 10 < angle < 80:

                right_angles.append(angle)

                cv2.line(
                    line_image,
                    (x1, y1),
                    (x2, y2),
                    255,
                    2
                )

    # =================================================
    # CALCULATE SPRAY ANGLE
    # =================================================

    spray_angle = None

    if left_angles and right_angles:

        left_mean = np.mean(left_angles)

        right_mean = np.mean(right_angles)

        spray_angle = 180 - abs(
            right_mean - left_mean
        )

    # =================================================
    # DISPLAY DETECTED BOUNDARIES
    # =================================================

    st.subheader("Detected Spray Boundary")

    st.image(
        line_image,
        use_container_width=True
    )

    # =================================================
    # DISPLAY ANGLE
    # =================================================

    if spray_angle is not None:

        st.metric(
            "Estimated Spray Angle",
            f"{spray_angle:.2f}°"
        )

    else:

        st.warning(
            "Spray angle could not be detected."
        )

    # =================================================
    # SPRAY DISTRIBUTION PROFILE
    # =================================================

    sample_y = int(h * 0.55)

    profile = gray[sample_y, :]

    fig, ax = plt.subplots(
        figsize=(12, 4)
    )

    ax.plot(profile)

    ax.set_title(
        "Spray Distribution Profile"
    )

    ax.set_xlabel(
        "Pixel Position"
    )

    ax.set_ylabel(
        "Intensity"
    )

    ax.grid(True)

    st.pyplot(fig)

    # =================================================
    # PEAK DETECTION
    # =================================================

    peaks, properties = find_peaks(
        profile,
        prominence=15,
        distance=20
    )

    fig2, ax2 = plt.subplots(
        figsize=(12, 4)
    )

    ax2.plot(profile)

    ax2.plot(
        peaks,
        profile[peaks],
        "rx",
        markersize=10
    )

    ax2.set_title(
        "Solid Stream Jet Detection"
    )

    ax2.set_xlabel(
        "Pixel Position"
    )

    ax2.set_ylabel(
        "Intensity"
    )

    ax2.grid(True)

    st.pyplot(fig2)

    st.metric(
        "Detected Peaks / Possible Jets",
        len(peaks)
    )

    # =================================================
    # SHOW BINARY MASK
    # =================================================

    st.subheader("Binary Spray Mask")

    st.image(
        thresh,
        use_container_width=True
    )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    "Built using Streamlit + OpenCV + SciPy"
)

