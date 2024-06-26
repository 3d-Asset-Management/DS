from PIL import Image
import io


class imgops:
    def __init__(self , contents , img_id):

        self.image_filepath = f"load/images/{img_id}.png"

        img = Image.open(io.BytesIO(contents))
        img.save(f"load/images/{img_id}.png")

    def upscale(self , params):
        pass





