import subprocess

class threestudio_dddgen():
    def __init__(self):
        pass

    def train(self , img_filepath):

        command = [
        "python",
        "launch.py",
        "--config", "configs/stable-zero123.yaml",
        "--train",
        "--gpu", "0",
        f"data.image_path={img_filepath}",
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