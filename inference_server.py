import time
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import base64
import uvicorn
import traceback
import numpy as np
import argparse

import torch as T
import torch.nn.functional as F
import torchaudio

import os
from typing import Optional

from utils import print_colored
from model import get_hertz_dev_config


argparse = argparse.ArgumentParser()

argparse.add_argument('--prompt_path', type=str, default='./prompts/bob_mono.wav', help="""
                      We highly recommend making your own prompt based on a conversation between you and another person.
                      bob_mono.wav seems to work better for two-channel than bob_stereo.wav.
                      """)
args = argparse.parse_args()


device = 'cuda' if T.cuda.is_available() else T.device('cpu')
print_colored(f"Using device: {device}", "grey")

model_config = get_hertz_dev_config(is_split=True)

model = model_config()
model = model.eval().bfloat16().to(device)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Hyperparams or something.
SAMPLE_RATE = 16000 # Don't change this
TEMPS = (0.8, (0.4, 0.1)) # You can change this, but there's also an endpoint for it.
REPLAY_SECONDS = 3 # What the user hears as context.

class AudioProcessor:
    def __init__(self, model, prompt_path):
        self.model = model
        self.prompt_path = prompt_path
        self.initialize_state(prompt_path)

    def initialize_state(self, prompt_path):
        loaded_audio, sr = torchaudio.load(prompt_path)
        self.replay_seconds = REPLAY_SECONDS
        
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            loaded_audio = resampler(loaded_audio)
            
        if loaded_audio.shape[0] == 1:
            loaded_audio = loaded_audio.repeat(2, 1)
            
        audio_length = loaded_audio.shape[-1] 
        num_chunks = audio_length // 2000
        loaded_audio = loaded_audio[..., :num_chunks * 2000]
            
        self.loaded_audio = loaded_audio.to(device)
            
        with T.autocast(device_type=device, dtype=T.bfloat16), T.inference_mode():
                self.model.init_cache(bsize=1, device=device, dtype=T.bfloat16, length=1024)
                self.next_model_audio = self.model.next_audio_from_audio(self.loaded_audio.unsqueeze(0), temps=TEMPS)
        self.prompt_buffer = None
        self.prompt_position = 0
        self.chunks_until_live = int(self.replay_seconds * 8)
        self.initialize_prompt_buffer()
        print_colored("AudioProcessor state initialized", "green")

    def initialize_prompt_buffer(self):
        self.recorded_audio = self.loaded_audio
        prompt_audio = self.loaded_audio.reshape(1, 2, -1)
        prompt_audio = prompt_audio[:, :, -(16000*self.replay_seconds):].cpu().numpy()
        prompt_audio_mono = prompt_audio.mean(axis=1)
        self.prompt_buffer = np.array_split(prompt_audio_mono[0], int(self.replay_seconds * 8))
        print_colored(f"Initialized prompt buffer with {len(self.prompt_buffer)} chunks", "grey")
    
    async def process_audio(self, audio_data):
        if self.chunks_until_live > 0:
            print_colored(f"Serving from prompt buffer, {self.chunks_until_live} chunks left", "grey")
            chunk = self.prompt_buffer[int(self.replay_seconds * 8) - self.chunks_until_live]
            self.chunks_until_live -= 1
            
            if self.chunks_until_live == 0:
                print_colored("Switching to live processing mode", "green")

            time.sleep(0.05)
            return chunk
        
        audio_tensor = T.from_numpy(audio_data).to(device)
        audio_tensor = audio_tensor.reshape(1, 1, -1)
        audio_tensor = T.cat([audio_tensor, self.next_model_audio], dim=1)
        
        with T.autocast(device_type=device, dtype=T.bfloat16), T.inference_mode():
            curr_model_audio = self.model.next_audio_from_audio(
                audio_tensor, 
                temps=TEMPS
            )
        print(f"Recorded audio shape {self.recorded_audio.shape}, audio tensor shape {audio_tensor.shape}")
        self.recorded_audio = T.cat([self.recorded_audio.cpu(), audio_tensor.squeeze(0).cpu()], dim=-1)

        self.next_model_audio = curr_model_audio

        return curr_model_audio.float().cpu().numpy()

    def cleanup(self):
        print_colored("Cleaning up audio processor...", "blue")
        os.makedirs('audio_recordings', exist_ok=True)
        torchaudio.save(f'audio_recordings/{time.strftime("%d-%H-%M")}.wav', self.recorded_audio.cpu(), SAMPLE_RATE)
        self.model.deinit_cache()
        self.initialize_state(self.prompt_path)
        print_colored("Audio processor cleanup complete", "green")

@app.post("/set_temperature")
async def set_temperature(token_temp: Optional[float] = None, categorical_temp: Optional[float] = None, gaussian_temp: Optional[float] = None):
    try:        
        global TEMPS
        TEMPS = (token_temp, (categorical_temp, gaussian_temp))
        
        print_colored(f"Temperature updated to: {TEMPS}", "green")
        return {"message": f"Temperature updated to: {TEMPS}", "status": "success"}
    except Exception as e:
        print_colored(f"Error setting temperature: {str(e)}", "red")
        return {"message": f"Error setting temperature: {str(e)}", "status": "error"}

@app.websocket("/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            audio_data = np.frombuffer(
                base64.b64decode(data.split(",")[1]),
                dtype=np.int16
            )
            audio_data = audio_data.astype(np.float32) / 32767.0
            processed_audio = await audio_processor.process_audio(audio_data)
            processed_audio = (processed_audio * 32767).astype(np.int16)
            
            processed_data = base64.b64encode(processed_audio.tobytes()).decode('utf-8')
            await websocket.send_text(f"data:audio/raw;base64,{processed_data}")
            
    except Exception as e:
            print_colored(f"WebSocket error: {e}", "red")
            print_colored(f"Full traceback:\n{traceback.format_exc()}", "red")
    finally:
        audio_processor.cleanup()
        await websocket.close()


audio_processor = AudioProcessor(model=model, prompt_path=args.prompt_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    print("Server started")
