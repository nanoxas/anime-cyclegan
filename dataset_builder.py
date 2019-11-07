from pathlib import Path
import numpy as np
from PIL import Image
import cv2


def read_faces(base_path_celeb, base_path_simpsons):
    count = 0
    celeb_faces = []
    simpsons_faces = []
    max_imgs = 500
    print('celeb_faces')
    for filename in Path(base_path_celeb).glob('**/*.jpg'):
        if count < max_imgs:
            try:
                img_orig = np.array(Image.open(filename))
                # print(img_orig)

                resized = cv2.resize(
                    img_orig, (64, 64), interpolation=cv2.INTER_AREA)

                if len(resized.shape) != 3:
                    continue

                resized = (resized / 255) * 2 - 1
                celeb_faces.append(resized)
                count += 1
            except Exception as e:
                print(e)
        else:
            break
    print('anime_faces')
    count = 0
    for filename in Path(base_path_simpsons).glob('**/*.png'):
        if count < max_imgs:
            try:
                img_orig = np.array(Image.open(filename))

                resized = cv2.resize(
                    img_orig, (64, 64), interpolation=cv2.INTER_AREA)

                if resized.shape[2] != 3:
                    print('grayscale_face')
                    continue

                resized = (resized / 255) * 2 - 1
                simpsons_faces.append(resized)
                count += 1
            except Exception as e:
                print(e)
        else:
            break

    return np.array(celeb_faces, dtype='float16'), np.array(
        simpsons_faces, dtype='float16')
