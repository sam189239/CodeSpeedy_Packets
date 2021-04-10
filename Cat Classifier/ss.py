import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import scipy
from scipy import misc
import imageio
from cat_classifier_utils import *

image_name = "external-content.duckduckgo.com.jpg" # enter name of the image and make sure it's in the same directory
label_y = [1] # (1 -> cat, 0 -> non-cat)
im = Image.open(r"external-content.duckduckgo.com.jpg")  
old_size = im.size
desired_size = 64
ratio = float(desired_size)/max(old_size)
new_size = tuple([int(x*ratio) for x in old_size])
im = im.resize(new_size, Image.ANTIALIAS)

new_im = Image.new("RGB", (desired_size, desired_size))
new_im.paste(im, ((desired_size-new_size[0])//2,
                    (desired_size-new_size[1])//2))
plt.imshow(new_im)
 
new_im = np.asarray(new_im)

new_im = np.reshape(new_im, (1, 12288))
new_im = new_im.flatten()
print(str(new_im.size))

new_im = new_im/255.
# new_im = Image.fromarray(np.transpose(new_im))
print(str(new_im.size))
#new_im = np.ndarray(np.transpose(new_im))

image = np.array(imageio.imread(image_name))
my_image = np.array(Image.fromarray(image).resize((64,64)))
my_image = my_image/255.

print(str(new_im.size))
print(str(my_image.size))

#plt.imshow(new_im)
plt.show()


# image = np.array(plt.imread(image_name))
# image = np.array(image.resize((64,64)))
# image = np.array(image.reshape((64*64*3,1)))


#np.array(Image.fromarray(image).resize((num_px*num_px*3,1)))
# my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
#image = image/255.
# plt.imshow(image)
# plt.show()


##################################################


#Try on your own image

# image_name = "external-content.duckduckgo.com.jpg" # enter name of the image and make sure it's in the same directory
# label_y = [1] # (1 -> cat, 0 -> non-cat)

# image = np.array(plt.imread(image_name))
# image = image.resize(size=(num_px,num_px))
# #np.array(Image.fromarray(image).resize((num_px*num_px*3,1)))
# # my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
# image = image/255.
# my_predicted_image = predict(image, label_y, parameters)

image_name = "external-content.duckduckgo.com.jpg" # enter name of the image and make sure it's in the same directory
label_y = [1] # (1 -> cat, 0 -> non-cat)
im = Image.open(r"external-content.duckduckgo.com.jpg")  
old_size = im.size
desired_size = 64
ratio = float(desired_size)/max(old_size)
new_size = tuple([int(x*ratio) for x in old_size])
im = im.resize(new_size, Image.ANTIALIAS)

new_im = Image.new("RGB", (desired_size, desired_size))
new_im.paste(im, ((desired_size-new_size[0])//2,
                    (desired_size-new_size[1])//2))
plt.imshow(new_im)
new_im = np.array(new_im)
np.reshape(new_im, (12288, 1))
new_im = new_im.flatten()
new_im = new_im/255.


image = np.array(imageio.imread(image_name))
my_image = np.array(Image.fromarray(image).resize((64,64)))
my_image = my_image/255.
my_predicted_image = predict(my_image, label_y, parameters)

plt.show()
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")