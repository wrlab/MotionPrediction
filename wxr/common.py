
MAX_NUM_ENVS = 32
SIM_DEVICE = 'cuda'
SIM_FPS = 36
TIME_STRIDE = 1 / SIM_FPS
TIME_WINDOW_SEC = 1
STATE_BUFFER_SIZE = SIM_FPS * TIME_WINDOW_SEC
MODEL_PATH = "model_15400.pt"