import numpy as np
import matplotlib.pyplot as plt
import cv2

def create_feathered_mask(width, height, side, feathering_size, save_path):
    """
    Create a mask with specified width and height where half of the mask is black and the other half is white.
    The border between the two halves is feathered.

    :param width: Width of the mask
    :param height: Height of the mask
    :param side: 'left' or 'right' for the white side
    :param feathering_size: Width of the feathered part
    :return: A mask as a numpy array
    """
    # Create a blank image
    mask = np.zeros((height, width))

    # Calculate the start and end points of the feathering
    if side == 'left':
        start, end = width // 2 - feathering_size // 2, width // 2 + feathering_size // 2
    else:
        start, end = width // 2 - feathering_size // 2, width // 2 + feathering_size // 2

    # Create the feathered edge
    for i in range(feathering_size):
        intensity = i / feathering_size
        mask[:, start + i] = 1 - intensity if side == 'left' else intensity

    # Fill in the solid parts of the mask
    mask[:, :start] = 1 if side == 'left' else 0
    mask[:, end:] = 0 if side == 'left' else 1
    # Save the mask to the specified path
    cv2.imwrite(save_path, mask * 255)

    return mask

# Example usage
width, height, side, feathering_size = 400, 200, 'left', 50
mask = create_feathered_mask(width, height, side, feathering_size, save_path='/home/kasra/AI_projects_/simple_diffusion/data/mask.png')

# Display the mask
plt.imshow(mask, cmap='gray')
plt.axis('off')
plt.show()
