import os
import re


def threestudio_savedir(base_dir , pattern = rf'^\[64, 128, 256\]_{id}\.png@.*?$'):

    for fn in os.listdir(base_dir):
    
        print(fn)
        match = re.match(pattern, fn)
    
        if match:
            current_path = os.path.join(base_dir , fn)
            print(current_path)
            
    save_dir = os.path.join(current_path , "save")

    return save_dir