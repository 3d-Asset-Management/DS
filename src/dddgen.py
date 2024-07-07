import subprocess
import re,os

class threestudio_dddgen():
    def __init__(self , img_filepath):

        self.img_filepath = img_filepath
        self.final_path = None

    def train(self):

        command = [
        "python",
        "launch.py",
        "--config", "configs/stable-zero123.yaml",
        "--train",
        "--gpu", "0",
        f"data.image_path={self.img_filepath}",
        "trainer.max_steps=100"
        ]

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Command executed successfully:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while executing command:\n{e.stderr}")

        base_dir = "outputs/zero123-sai/"

        return base_dir
    

    def extract_mesh(sel , save_dir):

        command = [
        "python",
        "launch.py",
        "--config", f"{save_dir}/../configs/parsed.yaml",
        "--export",
        "--gpu", "0",
        f"resume={save_dir}/../ckpts/last.ckpt",
        "system.exporter_type=mesh-exporter",
        "system.exporter.context_type=cuda",
        "system.geometry.isosurface_threshold=15.0"
    ]
    
        try:
            result = subprocess.run(command)
            print(f"Command executed successfully:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while executing command:\n{e.stderr}")

    def savedir(self , base_dir , pattern = rf'^\[64, 128, 256\]_{id}\.png@.*?$'):

        current_path = None
        for fn in os.listdir(base_dir):
        
            print(fn)
            match = re.match(pattern, fn)
        
            if match:
                current_path = os.path.join(base_dir , fn)
                print(current_path)
                
        save_dir = os.path.join(current_path , "save")

        return save_dir , current_path
    
    def renamedir(self , current_path , base_dir , id):

        final_path = os.path.join(base_dir , id)
        os.rename(current_path, final_path)
        self.final_path = final_path
        return final_path
