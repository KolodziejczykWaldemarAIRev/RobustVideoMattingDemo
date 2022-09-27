import subprocess
import threading
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
from PIL import Image
import cv2
import imutils


import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Tuple
from tqdm.auto import tqdm
from model import MattingNetwork

class VideoWriter:
    def __init__(self, path, frame_rate=25, bit_rate=1000000):
        self.container = av.open(path, mode='w')
        self.stream = self.container.add_stream('h264', rate=round(frame_rate))
        self.stream.pix_fmt = 'yuv420p'
        self.stream.bit_rate = bit_rate

    def write(self, frame):
        # frames: [H, W, C]
        self.stream.width = frame.shape[1]
        self.stream.height = frame.shape[0]
        frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        self.container.mux(self.stream.encode(frame))

    def close(self):
        self.container.mux(self.stream.encode())
        self.container.close()

def video_upload():
    content_file = st.sidebar.file_uploader("Choose a Content Video", type=["mp4"])

    if content_file is not None:
        vid_name = content_file.name
        with open(vid_name, mode='wb') as f:
            f.write(content_file.read())

        # writer = VideoWriter('out'+vid_name)

        video = cv2.VideoCapture(vid_name)
        total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        video.set(cv2.CAP_PROP_FPS, 25)

        frame_width = int(video.get(3))
        frame_height = int(video.get(4))
        writer = cv2.VideoWriter('out' + vid_name,
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 25.0, (frame_width, frame_height))

        progress_bar = st.progress(0)
        curr_frame = 0
        while True:
            success, image = video.read()
            # TODO process frame

            if not success:
                break
            curr_frame += 1
            writer.write(image)
            progress_bar.progress(curr_frame / total_frames)
        writer.release()

        convertedVideo = "output.mp4"
        st.subheader('Conversion of processed video to playable format...')
        subprocess.call(args=f"ffmpeg -y -i {'out'+vid_name} -c:v libx264 {convertedVideo}".split(" "))

        # video_file = open('out' + vid_name, 'rb')
        video_file = open(convertedVideo, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)


    else:
        st.warning("Upload an Image OR Untick the Upload Button)")
        st.stop()


def webcam_input(style_model_name):
    st.header("Webcam Live Feed")
    WIDTH = st.sidebar.select_slider('QUALITY (May reduce the speed)', list(range(150, 501, 50)))

    class NeuralStyleTransferTransformer(VideoTransformerBase):
        _width = WIDTH
        _model_name = style_model_name
        _model = None

        def __init__(self) -> None:
            self._model_lock = threading.Lock()

            self._width = WIDTH
            self._transform = transforms.ToTensor()
            self._device = 'cuda'
            self._model = None
            self._rec = [None] * 4
            self._downsample_ratio = None
            self._update_model()
            self._bgr = torch.tensor([120, 255, 155], device=self._device, dtype=torch.float32).div(255).view(1, 1, 3, 1, 1)

        def set_width(self, width):
            update_needed = self._width != width
            self._width = width
            if update_needed:
                self._update_model()

        def _update_model(self):
            variant = 'mobilenetv3'
            checkpoint = 'rvm_mobilenetv3.pth'
            with self._model_lock:
                self._model = MattingNetwork(variant).eval().to(self._device )
                self._model.load_state_dict(torch.load(checkpoint, map_location=self._device ))
                self._model = torch.jit.script(self._model)
                self._model = torch.jit.freeze(self._model)
                self._model = self._model.eval()
                self._rec = [None] * 4
                torch.set_grad_enabled(False)
                self._downsample_ratio = None

        def transform(self, frame):
            image = frame.to_ndarray(format="bgr24")

            if self._model == None:
                return image

            orig_h, orig_w = image.shape[:2]

            with self._model_lock:
                if self._downsample_ratio is None:
                    self._downsample_ratio = auto_downsample_ratio(orig_h, orig_w)


                image = self._transform(image)
                # print(image.shape)
                image = image.to(self._device, torch.float32, non_blocking=True).unsqueeze(0) # [B, T, C, H, W]
                # print(image.shape)
                fgr, pha, *self._rec = self._model(image, *self._rec, self._downsample_ratio)
                final_arr = fgr * pha + self._bgr * (1 - pha)
                # print(final_arr.shape)
                final_arr = final_arr[0].mul(255).byte().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
                # print(final_arr.shape)

            result = Image.fromarray(final_arr[0])
            return np.asarray(result.resize((orig_w, orig_h)))

    def auto_downsample_ratio(h, w):
        return min(512 / max(h, w), 1)

    ctx = webrtc_streamer(
        client_settings=ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        ),
        video_transformer_factory=NeuralStyleTransferTransformer,
        key="neural-style-transfer",
    )
    if ctx.video_transformer:
        ctx.video_transformer.set_width(WIDTH)
