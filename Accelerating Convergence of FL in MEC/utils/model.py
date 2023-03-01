import struct
import numpy as np
import json

state_item_number = 4
action_item_number = 10
reward_item_number = 1


def pack_state(state):
    tmp = bytes()
    for i in state:
        tmp = tmp + struct.pack("<f", i)
    return tmp


def unpack_states(data_):
    size_ = struct.calcsize("<f") * state_item_number
    start_ = 0
    states = []
    while start_ < len(data_):
        v_ = unpack_state(data_[start_:start_ + size_])
        states.append(v_)
        start_ += size_
    return np.array(states)


def unpack_state(data_):
    size_ = struct.calcsize("<f")
    start_ = 0
    state = []
    for i in range(state_item_number):
        v_ = struct.unpack("<f", data_[start_:start_ + size_])[0]
        state.append(v_)
        start_ += size_
    return np.array(state)


def pack_experience(state: np.ndarray, action: np.ndarray, reward: np.ndarray, state_next: np.ndarray):
    tmp = pack_state(state)

    for i in action:
        tmp = tmp + struct.pack("<f", i)

    for i in reward:
        tmp = tmp + struct.pack("<f", i)

    tmp = tmp + pack_state(state_next)
    return tmp


def unpack_experience(data_):
    size_ = struct.calcsize("<f")
    start_ = 0
    states = []
    actions = []
    reward = []
    states_next = []
    for i in range(state_item_number):
        v_ = struct.unpack("<f", data_[start_:start_ + size_])[0]
        states.append(v_)
        start_ += size_
    for i in range(action_item_number):
        v_ = struct.unpack("<f", data_[start_:start_ + size_])[0]
        actions.append(v_)
        start_ += size_
    for i in range(reward_item_number):
        v_ = struct.unpack("<f", data_[start_:start_ + size_])[0]
        reward.append(v_)
        start_ += size_
    for i in range(state_item_number):
        v_ = struct.unpack("<f", data_[start_:start_ + size_])[0]
        states_next.append(v_)
        start_ += size_
    states, actions, reward, states_next = np.array(states), np.array(actions), np.array(reward), np.array(states_next)
    return states, actions, reward, states_next


def pack_action(action: np.ndarray):
    tmp = bytes()
    for i in action:
        tmp = tmp + struct.pack("<f", i)
    return tmp


def unpack_actions(data_, n):
    size_ = struct.calcsize("<f") * action_item_number
    tmp = []
    start_ = 0
    for i in range(n):
        action_ = unpack_action(data_[start_:start_ + size_])
        tmp.append(action_)
        start_ += size_
    return np.array(tmp)


def unpack_action(data_):
    size_ = struct.calcsize("<f")
    start_ = 0
    tmp = []
    for i in range(action_item_number):
        v_ = struct.unpack("<f", data_[start_:start_ + size_])[0]
        tmp.append(v_)
        start_ += size_
    return np.array(tmp)


def parse_config(data: bytes, format_):
    decoder = None
    if format_ == "json":
        decoder = json.JSONDecoder()
    s_: str = data.decode()
    config_ = decoder.decode(s_)
    return config_


def pack_config(config: dict, format_):
    encoder = None
    if format_ == "json":
        encoder = json.JSONEncoder()
    r_ = encoder.encode(config)
    return r_.encode()


def format_write_dict(path, d: dict):
    f = open(path, "w")
    f.write("\n\nConfig = {\n")
    s_ = str(d)
    s_ = s_.strip("{}")
    tmp_ = s_.split(",")
    for i in range(len(tmp_)):
        t = tmp_[i]
        k, v = t.split(":")
        k = k.strip()
        v = v.strip()
        f.write("    ")
        f.write(k)
        f.write(": ")
        f.write(v)
        if i != (len(tmp_) - 1):
            f.write(",\n")
    f.write("\n}")
    f.close()
