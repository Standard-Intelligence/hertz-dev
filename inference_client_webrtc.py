# server.py remains the same as before

# Updated client.py
import asyncio
import websockets
import numpy as np
import base64
import argparse
import requests
import time
import torch
import torchaudio
import pydub

import av
import streamlit as st
from typing import List
from streamlit_webrtc import WebRtcMode, webrtc_streamer

class AudioClient:
    def __init__(self, server_url="ws://localhost:8000", token_temp=None, categorical_temp=None, gaussian_temp=None):
        # Convert ws:// to http:// for the base URL
        self.base_url = server_url.replace("ws://", "http://")
        self.server_url = f"{server_url}/audio"
        
        # Set temperatures if provided
        if any(t is not None for t in [token_temp, categorical_temp, gaussian_temp]):
            response_message = self.set_temperature_and_echo(token_temp, categorical_temp, gaussian_temp)
            print(response_message)

        self.downsampler = torchaudio.transforms.Resample(STREAMING_SAMPLE_RATE, SAMPLE_RATE)
        self.upsampler = torchaudio.transforms.Resample(SAMPLE_RATE, STREAMING_SAMPLE_RATE)
        self.ws = None
        self.in_buffer = None
        self.out_buffer = None
    
    def set_temperature_and_echo(self, token_temp=None, categorical_temp=None, gaussian_temp=None, echo_testing = False):
        """Send temperature settings to server"""
        params = {}
        if token_temp is not None:
            params['token_temp'] = token_temp
        if categorical_temp is not None:
            params['categorical_temp'] = categorical_temp
        if gaussian_temp is not None:
            params['gaussian_temp'] = gaussian_temp
            
        response = requests.post(f"{self.base_url}/set_temperature", params=params)
        response_message = response.json()['message']
        return response_message
    
    def _resample(self, audio_data: np.ndarray, resampler: torchaudio.transforms.Resample) -> np.ndarray:
        audio_data = audio_data.astype(np.float32) / 32767.0
        audio_data = resampler(torch.tensor(audio_data)).numpy()
        audio_data = (audio_data * 32767.0).astype(np.int16)
        return audio_data
    
    def upsample(self, audio_data: np.ndarray) -> np.ndarray:
        return self._resample(audio_data, self.upsampler)
    
    def downsample(self, audio_data: np.ndarray) -> np.ndarray:
        return self._resample(audio_data, self.downsampler)

    async def process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        if self.ws is None:
            self.ws = await websockets.connect(self.server_url)

        audio_data = audio_data.reshape(-1, CHANNELS)
        print(f'Data from microphone:{audio_data.shape, audio_data.dtype, audio_data.min(), audio_data.max()}')

        # Convert to base64
        audio_b64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
        
        # Send to server
        time_sent = time.time()
        await self.ws.send(f"data:audio/raw;base64,{audio_b64}")
    
        # Receive processed audio
        response = await self.ws.recv()
        response = response.split(",")[1]
        time_received = time.time()
        print(f"Data sent: {audio_b64[:10]}. Data received: {response[:10]}. Received in {(time_received - time_sent) * 1000:.2f} ms")
        processed_audio = np.frombuffer(
            base64.b64decode(response),
            dtype=np.int16
        ).reshape(-1, CHANNELS)
        print(f'Data from model:{processed_audio.shape, processed_audio.dtype, processed_audio.min(), processed_audio.max()}')

        if CHANNELS == 1:
            processed_audio = processed_audio.reshape(-1)
        return processed_audio
    
    async def queued_audio_frames_callback(self, frames: List[av.AudioFrame]) -> List[av.AudioFrame]:
        out_frames = []
        for frame in frames:
            # Read in audio and convert to mono
            audio_data = frame.to_ndarray()
            sound = pydub.AudioSegment(
                data=audio_data.tobytes(),
                sample_width=frame.format.bytes,
                frame_rate=frame.sample_rate,
                channels=len(frame.layout.channels),
            )
            sound = sound.set_channels(CHANNELS)
            audio_data = np.array(sound.get_array_of_samples())

            # Downsample audio
            audio_data = self.downsample(audio_data)

            # Add audio to input buffer
            if self.in_buffer is None:
                self.in_buffer = audio_data
            else:
                self.in_buffer = np.concatenate((self.in_buffer, audio_data), axis=-1)
            
            # Take BLOCK_SIZE samples from input buffer if available for processing
            if self.in_buffer.shape[0] >= BLOCK_SIZE:
                audio_data = self.in_buffer[:BLOCK_SIZE]
                self.in_buffer = self.in_buffer[BLOCK_SIZE:]
            else:
                audio_data = None
            
            # Process audio if available and add resulting audio to output buffer
            if audio_data is not None:
                audio_data = await self.process_audio(audio_data)
                if self.out_buffer is None:
                    self.out_buffer = audio_data
                else:
                    self.out_buffer = np.concatenate((self.out_buffer, audio_data), axis=-1)

            # Take `out_samples` samples from output buffer if available for output
            out_samples = int(frame.samples * SAMPLE_RATE / STREAMING_SAMPLE_RATE)
            if self.out_buffer is not None and self.out_buffer.shape[0] >= out_samples:
                audio_data = self.out_buffer[:out_samples]
                self.out_buffer = self.out_buffer[out_samples:]
            else:
                audio_data = None

            # Output silence if no audio data available
            if audio_data is None:
                # output silence
                audio_data = np.zeros(320, dtype=np.int16)
            
            # Upsample output audio
            audio_data = self.upsample(audio_data)

            # Repeat audio data to match the original number of channels
            audio_data = np.repeat(audio_data, 2).reshape(1, -1)

            # return audio data as AudioFrame
            new_frame = av.AudioFrame.from_ndarray(audio_data, format=frame.format.name, layout=frame.layout.name)
            new_frame.sample_rate = frame.sample_rate
            out_frames.append(new_frame)

        return out_frames

    def stop(self): 
        if self.ws is not None:
            # TODO: this hangs. Figure out why.
            #asyncio.get_event_loop().run_until_complete(self.ws.close())
            print("Websocket closed")
        self.ws = None
        self.in_buffer = None
        self.out_buffer = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio Client with Temperature Control')
    parser.add_argument('--token_temp', '-t1', type=float, help='Token (LM) temperature parameter')
    parser.add_argument('--categorical_temp', '-t2', type=float, help='Categorical (VAE) temperature parameter')
    parser.add_argument('--gaussian_temp', '-t3', type=float, help='Gaussian (VAE) temperature parameter')
    parser.add_argument('--server', '-s', default="ws://localhost:8000", 
                        help='Server URL (default: ws://localhost:8000)')
    parser.add_argument("--use_ice_servers", action="store_true", help="Use public STUN servers")
    
    args = parser.parse_args()
    
    # Audio settings
    STREAMING_SAMPLE_RATE = 48000
    SAMPLE_RATE = 16000
    BLOCK_SIZE = 2000
    CHANNELS = 1
    
    audio_client = st.session_state.get("audio_client")
    if audio_client is None:
        audio_client = AudioClient(
            server_url=args.server,
            token_temp=args.token_temp,
            categorical_temp=args.categorical_temp,
            gaussian_temp=args.gaussian_temp
        )
        st.session_state.audio_client = audio_client

    with st.sidebar:
        token_temp_default = args.token_temp if args.token_temp is not None else 0.8
        token_temp = st.slider("Token Temperature", 0.05, 2.0, token_temp_default, step=0.05)
        categorical_temp_default = args.categorical_temp if args.categorical_temp is not None else 0.4
        categorical_temp = st.slider("Categorical Temperature", 0.01, 1.0, categorical_temp_default, step=0.01)
        gaussian_temp_default = args.gaussian_temp if args.gaussian_temp is not None else 0.1
        gaussian_temp = st.slider("Gaussian Temperature", 0.01, 1.0, gaussian_temp_default, step=0.01)
        if st.button("Set Temperatures"):
            response_message = audio_client.set_temperature_and_echo(token_temp, categorical_temp, gaussian_temp)
            st.write(response_message)

    # Use a free STUN server from Google if --use_ice_servers is given
    # (found in get_ice_servers() at https://github.com/whitphx/streamlit-webrtc/blob/main/sample_utils/turn.py)
    rtc_configuration = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]} if args.use_ice_servers else None
    webrtc_streamer(
        key="streamer",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"audio": True, "video": False},
        queued_audio_frames_callback=audio_client.queued_audio_frames_callback,
        on_audio_ended=audio_client.stop,
        async_processing=True,
    )
        
