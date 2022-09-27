import streamlit as st

from input import webcam_input, video_upload

st.title("Background removal")
st.sidebar.title('Navigation')
method = st.sidebar.radio('Go To ->', options=['Webcam', 'Upload'])
st.sidebar.header('Options')

if method == 'Upload':
    video_upload()
else:
    webcam_input('Composition_vii')

chosen_scaling_factor = st.sidebar.selectbox("Choose the scaling factor: ", [0.125, 0.2])