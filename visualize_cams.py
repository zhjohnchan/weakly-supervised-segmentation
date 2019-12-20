import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def getCAM(image, weights, display=False):
    # Normalize
    cam = (weights - weights.min()) / (weights.max() - weights.min())
    # Resize as image size
    cam_resize = cv2.resize(cam, (256, 256))
    # Format as CV_8UC1 (as applyColorMap required)
    cam_resize = 255 * cam_resize
    cam_resize = cam_resize.astype(np.uint8)
    # Get Heatmap
    heatmap = cv2.applyColorMap(cam_resize, cv2.COLORMAP_JET)
    # Zero out
    # heatmap[np.where(cam_resize <= 100)] = 0
    out = cv2.addWeighted(src1=image, alpha=0.8, src2=heatmap, beta=0.4, gamma=0)

    if display:
        plt.imshow(out[:, :, ::-1])
        plt.show()
    return out


if __name__ == '__main__':
    cams_name = 'saved/kidney_bc/cams/10946.npy'
    image_name = 'saved/kidney_bc/images/10946.png'
    mask_name = 'saved/kidney_bc/masks/10946.png'

    cam = np.load(os.path.join(cams_name))
    image = cv2.imread(image_name)
    mask = cv2.imread(mask_name)
    mask = cv2.resize(mask, (256, 256))

    norm_cam = [cam for key, cam in cam.item().items()]
    norm_cam = np.array(norm_cam)

    image_with_cam = getCAM(image, norm_cam[0])
    cv2.imwrite('saved/image_with_cam_1.png', image_with_cam)

    image_with_cam = getCAM(image, norm_cam[1])
    cv2.imwrite('saved/image_with_cam_2.png', image_with_cam)

    image_with_cam = getCAM(image, norm_cam[0] + norm_cam[1])
    cv2.imwrite('saved/image_with_cam_3.png', image_with_cam)

    image_with_cam = getCAM(mask, norm_cam[0] + norm_cam[1])
    cv2.imwrite('saved/mask_with_cam.png', image_with_cam)

    print('Complete Generating the figures we use in the presentation.')