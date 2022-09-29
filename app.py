import streamlit as st

from input import webcam_input, video_upload

st.sidebar.image('https://png.pngitem.com/pimgs/s/207-2073499_translate-platform-from-english-to-spanish-work-in.png',
 caption=None,
  width=200,
   use_column_width=None,
    clamp=False,
     channels="RGB",
      output_format="auto")


st.title("Background removal")
st.sidebar.title('Navigation')
method = st.sidebar.radio('Input type:', options=['Webcam', 'Upload (BONUS!!!)'])
st.sidebar.header('Options')


if method == 'Upload (BONUS!!!)':
    video_upload()
else:
    webcam_input()

