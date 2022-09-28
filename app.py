import streamlit as st

from input import webcam_input, video_upload

st.title("Background removal")
st.sidebar.title('Navigation')
method = st.sidebar.radio('Input type:', options=['Webcam', 'Upload'])
st.sidebar.header('Options')


if method == 'Upload':
    video_upload()
else:
    webcam_input()

