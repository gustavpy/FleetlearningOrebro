import sys

sys.path.append('../fleet-learning-young-talents')

from zod import ZodFrames
from PIL import Image

NO_CLIENTS = 40

def main() -> None:
    zod_frames = ZodFrames("/mnt/ZOD", version="full")

    #Enter frame_ids here and then run the code to get images
    bilder = [96324, 61929]
    for i in range(len(bilder)):
        frame = zod_frames[bilder[i]]
        pixel_data = frame.get_image()  
        image = Image.fromarray(pixel_data)

        # Save the image as a PNG
        image.save(f'Presentation_bild{i}.png', format='PNG')  # Replace 'output.png' with the desired output file name

main()