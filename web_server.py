import numpy as np
import threading

from wxr.common import *
from wxr.util import *

from wxr.gym_runner import GymRunner

from isaacgymenvs.utils.torch_jit_utils import quat_mul

import torch

async_mode = None

from flask import Flask, render_template
import socketio

sio = socketio.Server(
    async_mode=async_mode,
    cors_allowed_origins=[
        'http://192.168.1.183:8000',
        'https://admin.socket.io',
    ])

app = Flask(__name__)
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)
app.config['SECRET_KEY'] = 'secret'
thread = None
gym = GymRunner(MAX_NUM_ENVS, MODEL_PATH, 'cuda')

# server buffers
assign_table = np.arange(MAX_NUM_ENVS)
env_id_table = {}
ass_idx = 0

# gym buffers
head_state_buf = torch.zeros((STATE_BUFFER_SIZE, MAX_NUM_ENVS, 7), dtype=torch.float32, device='cpu')
left_hand_state_buf = torch.zeros((STATE_BUFFER_SIZE, MAX_NUM_ENVS, 3), dtype=torch.float32, device='cpu')
right_hand_state_buf = torch.zeros((STATE_BUFFER_SIZE, MAX_NUM_ENVS, 3), dtype=torch.float32, device='cpu')

# initilize buffers to valid values
def init_buffers(env_ids):
    head_state_buf[:,env_ids,2] = 1.4     # default head height
    head_state_buf[:,env_ids,6] = 1.0     # make quaternion to valid value
    left_hand_state_buf[:,env_ids,1] = 0.18       # leftward is +y ,it looks at +x axis
    left_hand_state_buf[:,env_ids,2] = 0.83 - 1.4  # default hand height, this buffer represent head's coordinate
    right_hand_state_buf[:,env_ids,1] = -0.18     # rightward is -y ,it looks at +x axis
    right_hand_state_buf[:,env_ids,2] = 0.83 - 1.4 # default hand height, this buffer represent head's coordinate
    gym.reset_buf[env_ids] = 2
init_buffers(torch.tensor([x for x in range(MAX_NUM_ENVS)], dtype=torch.long, device='cpu'))

head_updated_idx = -1
lhand_updated_idx = -1
rhand_updated_idx = -1

buffer_lock = threading.Lock()

def background_thread():
    while True:
        sio.sleep()
        # copy state buffers
        buffer_lock.acquire()
        hsb = torch.clone(head_state_buf)
        lhb = torch.clone(left_hand_state_buf)
        rhb = torch.clone(right_hand_state_buf)
        buffer_lock.release()
        
        # step simulation
        step_ret = gym.step(hsb, lhb, rhb)
        
        if not step_ret:
            continue
        else:
            root_state, link_state = step_ret
            
        sio.sleep()
        
        def to_wxr_skeleton(quat, base_trans_euler=[0.0, 0.0, 0.0], degree=True):
            base = np.array(base_trans_euler) / 180.0 * np.pi if degree else np.array(base_trans_euler)
            base_quat = torch.Tensor(euler_to_quat(base))
            quat = quat_mul(quat, base_quat)
            return isaac_to_wxr_quat(quat.numpy()).astype(np.float32)
            
        buffer_lock.acquire()
        env_id_table_buf = env_id_table.copy()
        buffer_lock.release()
        for (sessionName, env_idx) in env_id_table_buf.items():
            #root            = to_wxr_skeleton(link_state[env_idx,0,3:7])
            root            = to_wxr_skeleton(root_state[env_idx, 3:7], [180,180,0])
            torso           = to_wxr_skeleton(link_state[env_idx,1,3:7], [180,180,0])
            neck            = to_wxr_skeleton(link_state[env_idx,2,3:7], [180,180,0])
            right_upper_arm = to_wxr_skeleton(link_state[env_idx,3,3:7], [0,180,0])
            right_lower_arm = to_wxr_skeleton(link_state[env_idx,4,3:7], [0,180,0])
            right_hand      = to_wxr_skeleton(link_state[env_idx,5,3:7], [0,180,0])
            left_upper_arm  = to_wxr_skeleton(link_state[env_idx,6,3:7], [0,180,0])
            left_lower_arm  = to_wxr_skeleton(link_state[env_idx,7,3:7], [0,180,0])
            left_hand       = to_wxr_skeleton(link_state[env_idx,8,3:7], [0,180,0])
            right_thigh     = to_wxr_skeleton(link_state[env_idx,9,3:7], [0,180,0])
            right_shin      = to_wxr_skeleton(link_state[env_idx,10,3:7], [0,180,0])
            right_foot      = to_wxr_skeleton(link_state[env_idx,11,3:7], [0,120,0])
            left_thigh      = to_wxr_skeleton(link_state[env_idx,12,3:7], [0,180,0])
            left_shin       = to_wxr_skeleton(link_state[env_idx,13,3:7], [0,180,0])
            left_foot       = to_wxr_skeleton(link_state[env_idx,14,3:7], [0,120,0])
            head            = neck
            quat_arr = bytes(np.concatenate((root, right_thigh, right_shin, right_foot, 
                                            left_thigh, left_shin, left_foot, torso, 
                                            left_upper_arm, left_lower_arm, neck, head, 
                                            right_upper_arm, right_lower_arm), axis=0))
            
            root_pos = list(isaac_to_wxr(root_state[0,0:3].numpy()).astype(float))
            head_pos = list(isaac_to_wxr(link_state[0,2,0:3].numpy()).astype(float))
            
            skeletonData = {
                'quatArr' : quat_arr,
                'rootPos' : root_pos,
                'headPos' : head_pos,
                'bodypart' : 'body',
            }
            
            # emit skeleton data to wxr
            sio.emit('vrMotionPredBodyMoving', {'id': sessionName, 'data': skeletonData})

@sio.event
def user_msg(sid, data):
    print("sid: ", sid)
    print("data: ", data)
    sio.emit('message', data, sid)   # client only transmit
    
@sio.event
def all_msg(sid, data):
    print("sid: ", sid)
    print("data: ", data)
    sio.emit('message', data)   # broadcase transmit
    
@sio.event
def connect(sid, environ, auth):
    print('connect', sid)
    
@sio.event
def disconnect(sid):
    print('disconnect', sid)
    
@sio.on('join')
def connect(sid, sessionName):
    global ass_idx
    print('join ', sessionName)
    
    buffer_lock.acquire()
    env_id = assign_table[ass_idx]
    env_id_table[sessionName] = env_id
    ass_idx += 1
    buffer_lock.release()
    
@sio.on('quit')
def disconnect(sid, sessionName):
    global ass_idx
    print('quit', sessionName)
    
    buffer_lock.acquire()
    env_id = env_id_table[sessionName]
    del(env_id_table[sessionName])
    assign_table[ass_idx] = env_id
    ass_idx -= 1
    init_buffers(env_id)
    buffer_lock.release()
    
@sio.on('userHeadSensorData')
def get_head_state(sid, sdata):
    sessionName = sdata['id']
    sensorData = sdata['data']
    # sensor datas
    head_pos = sensorData['pos']
    head_rot = sensorData['rot']
    head_pos = wxr_to_isaac(head_pos)
    head_rot = wxr_to_isaac(head_rot)
    
    ### lock ###
    buffer_lock.acquire()
    # find user's environment id and buffer's curruent time index
    if sessionName not in env_id_table:
        return
    env_id = env_id_table[sessionName]
    idx = get_curr_idx()
        
    # fill buffers
    if (gym.reset_buf[env_id] == 2):
        gym.lock.acquire()
        gym.head_offset[env_id] = 1.4 - head_pos[2]
        gym.height_scale[env_id] = 1.4 / 1.8
        gym.reset_idx_buf[env_id] = idx
        gym.reset_buf[env_id] = 1
        gym.lock.release()
    # position
    head_state_buf[idx, env_id, 0:3] = torch.Tensor(head_pos) + gym.head_offset[env_id]
    head_state_buf[idx, env_id, 3:7] = torch.Tensor(euler_to_quat(head_rot))
    
    gym.head_update_time = round_to_sliced_time(time.time())
    buffer_lock.release()
    ### lock ###
    
@sio.on('userHandSensorData')
def get_hand_state(sid, sdata):
    sessionName = sdata['id']
    sensorData = sdata['data']
    # sensor datas
    valid = sensorData['valid']
    handPoses = sensorData['pos']
    left_hand_valid = valid['left']
    right_hand_valid = valid['right']
    
    left_hand_pos = handPoses['left']
    right_hand_pos = handPoses['right']
    left_hand_pos = wxr_to_isaac(left_hand_pos)
    right_hand_pos = wxr_to_isaac(right_hand_pos)
    
    ### lock ###
    buffer_lock.acquire()
    # find user's environment id and buffer's curruent time index
    if sessionName not in env_id_table:
        return
    env_id = env_id_table[sessionName]
    idx = get_curr_idx()

    # fill buffers
    if left_hand_valid:
        left_hand_state_buf[idx, env_id, 0:3] = torch.Tensor(left_hand_pos)
    if right_hand_valid:
        right_hand_state_buf[idx, env_id, 0:3] = torch.Tensor(right_hand_pos)
    
    gym.lhand_update_time = round_to_sliced_time(time.time())
    gym.rhand_update_time = round_to_sliced_time(time.time())
    buffer_lock.release()
    ### lock ###
    
if __name__ == '__main__':
    if sio.async_mode == 'threading':
        thread = sio.start_background_task(background_thread)
        app.run(host='192.168.1.183', port=8000, threaded=True)