# hertz-dev

Hertz-dev is an open-source, first-of-its-kind base model for full-duplex conversational audio.

See our blog post for more details: https://si.inc/hertz-dev/

## Setup

Inference is known to work on Python 3.10 and CUDA 12.1. Other versions have not been tested as thoroughly. If you want to use CUDA 12.1, you'll need to install torch with `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` before running `pip install -r requirements.txt`.

All three scripts will automatically download the models to the `./ckpt` directory, and checkpoints are also accessible at https://ckpt.si.inc/hertz-dev/index.txt

## Usage

We recommend starting by using `inference.ipynb` to generate one- or two-channel completions from a prompt.

Then, you can use `inference_client.py` and `inference_server.py` to talk to the model live through your microphone.
These are currently experimental, and have primarily been tested with Ubuntu on the server and MacOS on the client.

Alternatively, you can use `inference_client_webrtc.py`, which is built on [streamlit](https://streamlit.io/) + [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) and runs in a browser:
```bash
# Install additional requirements
pip install -r requirements_webrtc.txt
# Run the client
streamlit run inference_client_webrtc.py
```