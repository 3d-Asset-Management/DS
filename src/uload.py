import subprocess

class s3_uload():
    def __init__(self , s3_link):

        self.s3_link = s3_link

        result = subprocess.run(
            ['aws', 's3', 'cp', 'model_path', s3_link, '--recursive'],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
