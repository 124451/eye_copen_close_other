from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
def main():
    img = Image.open(r"E:\Coding\headpose\headpose\data\300W_LP\AFW\AFW_261068_1_6.jpg")
    plt.subplot(121)
    plt.imshow(img)
    npimg = np.uint8(img)
    print(img.getpixel((0,0)))
    print("model:{}".format(img.mode))
    rgb_img = img.convert('RGB')
    print("model:{}".format(rgb_img.mode))
    plt.subplot(122)
    plt.imshow(rgb_img)
    nprgb = np.array(rgb_img)
    plt.show()
    print("dsa")




if __name__ == "__main__":
    main()