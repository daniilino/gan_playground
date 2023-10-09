import os
import cv2
import time

import torch
import numpy as np

def normalize_image(image):
    dtype = image.dtype
    image = image.astype(float)
    image = image - np.min(image)
    image = image / np.max(image) * 255
    image = image.astype(dtype)
    return image

def interpolate_two_points(z_1, z_2, num_steps, generator_fx, export_imgs_to=""):

    if export_imgs_to != "" and not os.path.exists(export_imgs_to):
        os.makedirs(export_imgs_to)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z_1 = z_1.to(device)
    z_2 = z_2.to(device)

    line = z_2 - z_1
    coefs = torch.arange(0, 1+(1/num_steps), 1/num_steps, dtype=torch.float).to(device)
    zs = z_1[None, :] + (line[None, :] * coefs[:, None])

    generator_fx.to(device)
    generator_fx.eval()
    with torch.no_grad():
        generations = generator_fx(zs).permute(0, 2, 3, 1).detach().cpu().numpy()
    
    imgs = []
    for i, g in enumerate(generations):
        img = cv2.resize(g, (1200,600), interpolation = cv2.INTER_AREA)
        if export_imgs_to != "":
            cv2.imwrite(os.path.join(export_imgs_to, f"gen{i:05d}.png"), (normalize_image(img)).astype(np.uint8))
        imgs.append(img)

    for img in imgs:
        cv2.imshow("inteqrpolation", img)
        time.sleep(0.005)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()