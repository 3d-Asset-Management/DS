from PIL import Image
import io

class ImgOps:
    def __init__(self , contents , img_id):

        self.img_filepath = f"load/images/{img_id}.png"
        self.img = None
        self.img_id = img_id
        print(img_id)

        img = Image.open(io.BytesIO(contents))
        img.save(f"load/images/{img_id}.png")
        print(self.img_filepath)
        self.img = img

    def upscale(self , params):
        pass





