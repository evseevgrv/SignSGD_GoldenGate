from collections import OrderedDict

def get_new_key(key):
    # Split the key into parts
    parts = key.split('.')
    # Remove '_orig_mod' from parts
    parts = [part for part in parts if part != '_orig_mod']
    # Join parts back together
    new_key = '.'.join(parts)
    return new_key

def transform_state_dict_keys(state_dict):
    new_state_dict = OrderedDict()
    
    for key in state_dict:
        new_key = get_new_key(key)
        new_state_dict[new_key] = state_dict[key]
    
    return new_state_dict

def transform_list(list):
    new_list = []
    for key in list:
        new_key = get_new_key(key)
        new_list.append(new_key)
    return new_list

def transform_optimizer_state_dict(optimizer_state_dict):
    optimizer_state_dict["state"] = transform_state_dict_keys(optimizer_state_dict["state"])
    for group in optimizer_state_dict["param_groups"]:
        group["params"] = transform_list(group["params"])
    return optimizer_state_dict